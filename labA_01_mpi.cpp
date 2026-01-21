#include <chrono>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <random>
#include <vector>

double lower_bound = -1000.0;
double upper_bound = 1000.0;
std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
std::default_random_engine re; // NOLINT(cert-msc51-cpp)

double random_e() { return unif(re); }

void generate(std::vector<double> *v, double (*gen)()) {
  for (double &i : *v) {
    i = gen();
  }
}

// Compute local portion of CE (row sums)
void compute_CE_local(int n, int local_rows, std::vector<double> &C_local,
                      std::vector<double> &CE_local) {
  for (int i = 0; i < local_rows; i++) {
    double tmp = 0.0;
    for (int k = 0; k < n; k++) {
      tmp += C_local[i * n + k];
    }
    CE_local[i] = tmp;
  }
}

// Compute local contribution to trace
double compute_trace_local(int n, int local_rows, std::vector<double> &B_local,
                           std::vector<double> &CE) {
  double trace = 0.0;
  for (int i = 0; i < local_rows; i++) {
    double tmp = 0.0;
    for (int k = 0; k < n; k++) {
      tmp += B_local[i * n + k] * CE[k];
    }
    trace += tmp;
  }
  return trace;
}

// Compute local contribution to z1
double compute_z1_local(int n, int local_rows, int row_offset,
                        std::vector<double> &B_local, double Ex,
                        std::vector<double> &y) {
  double z1 = 0.0;
  for (int i = 0; i < local_rows; i++) {
    int global_i = row_offset + i;
    double tmp = 0.0;
    for (int j = 0; j < n; j++) {
      tmp += B_local[i * n + j] * Ex * y[global_i];
    }
    z1 += tmp;
  }
  return z1;
}

double compute_z2(int n, std::vector<double> &x, std::vector<double> &y) {
  double z2 = 0.0;
  for (int i = 0; i < n; i++) {
    z2 += x[i] * y[i];
  }
  return z2;
}

// Compute local portion of A
void compute_A_local(int n, int local_rows, int row_offset,
                     std::vector<double> &A_local, std::vector<double> &C_local,
                     double trace, double z) {
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < n; j++) {
      A_local[i * n + j] = trace * C_local[i * n + j] + z;
    }
  }

  // Add 1 to diagonal elements (for rows owned by this process)
  for (int i = 0; i < local_rows; i++) {
    int global_i = row_offset + i;
    A_local[i * n + global_i]++;
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc > 64) {
    if (rank == 0)
      std::cerr << "too many input parameters!" << std::endl;
    MPI_Finalize();
    return 1;
  }

  const std::vector<std::string> args(argv + 1, argv + argc);

  int n = -1;
  bool print_result = false;
  bool is_debug = false;
  bool is_fixed_seed = false;

  for (const auto &arg : args) {
    if (arg == "--help") {
      if (rank == 0) {
        std::cout << "Usage: " << argv[0] << " [--help] [--print] n"
                  << std::endl;
      }
      MPI_Finalize();
      return 0;
    }
    if (arg == "--print") {
      print_result = true;
      continue;
    }
    if (arg == "--debug") {
      is_debug = true;
      print_result = true;
      n = 2;
      break;
    }
    if (arg == "--fixed-seed") {
      is_fixed_seed = true;
      continue;
    }
    n = std::stoi(arg);
  }

  // Calculate row distribution across processes
  int local_rows = n / size;
  int remainder = n % size;
  if (rank < remainder) {
    local_rows++;
  }
  int row_offset = rank * (n / size) + std::min(rank, remainder);

  // Vectors needed by all processes (full copies)
  std::vector<double> x(n);
  std::vector<double> y(n);
  std::vector<double> CE(n);

  // Local portions of matrices
  std::vector<double> B_local(local_rows * n);
  std::vector<double> C_local(local_rows * n);
  std::vector<double> A_local(local_rows * n);
  std::vector<double> CE_local(local_rows);

  // Full matrices only on root
  std::vector<double> B, C, A;

  if (rank == 0) {
    B.resize(n * n);
    C.resize(n * n);
    A.resize(n * n);

    if (is_fixed_seed) {
      re.seed(42); // NOLINT(cert-msc51-cpp)
    } else {
      re.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    if (is_debug) {
      B.assign({1, 2, 3, 4});
      C.assign({5, 6, 7, 8});
      x.assign({2, 3});
      y.assign({4, 5});
    } else {
      generate(&B, random_e);
      generate(&C, random_e);
      generate(&x, random_e);
      generate(&y, random_e);
    }
  }

  // Broadcast x and y to all processes
  MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(y.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Prepare counts and displacements for Scatterv/Gatherv
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  std::vector<int> recvcounts(size);
  std::vector<int> recvdispls(size);

  for (int i = 0; i < size; i++) {
    int rows_i = n / size + (i < remainder ? 1 : 0);
    sendcounts[i] = rows_i * n;
    recvcounts[i] = rows_i;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    recvdispls[i] = (i == 0) ? 0 : recvdispls[i - 1] + recvcounts[i - 1];
  }

  // Scatter B and C matrices (row-wise distribution)
  MPI_Scatterv(rank == 0 ? B.data() : nullptr, sendcounts.data(), displs.data(),
               MPI_DOUBLE, B_local.data(), local_rows * n, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
  MPI_Scatterv(rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(),
               MPI_DOUBLE, C_local.data(), local_rows * n, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::system_clock::time_point start_time =
      std::chrono::system_clock::now();

  // Compute Ex = sum of x (all processes have full x)
  double Ex = std::accumulate(x.begin(), x.end(), 0.0);

  // Compute local CE (row sums)
  compute_CE_local(n, local_rows, C_local, CE_local);

  // Gather CE to all processes (needed for trace computation)
  MPI_Allgatherv(CE_local.data(), local_rows, MPI_DOUBLE, CE.data(),
                 recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  // Compute trace with reduction
  double local_trace = compute_trace_local(n, local_rows, B_local, CE);
  double trace;
  MPI_Allreduce(&local_trace, &trace, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Compute z1 with reduction
  double local_z1 = compute_z1_local(n, local_rows, row_offset, B_local, Ex, y);
  double z1;
  MPI_Allreduce(&local_z1, &z1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Compute z2 (all processes have full x and y)
  double z2 = compute_z2(n, x, y);

  double z = z1 / z2;

  // Compute local portion of A
  compute_A_local(n, local_rows, row_offset, A_local, C_local, trace, z);

  // Gather A to root
  MPI_Gatherv(A_local.data(), local_rows * n, MPI_DOUBLE,
              rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::milliseconds elapsed_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now() - start_time);

  if (rank == 0) {
    if (print_result) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          std::cout << std::setprecision(10) << A[i * n + j] << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << elapsed_milliseconds.count() << std::endl;
  }

  MPI_Finalize();
  return 0;
}
