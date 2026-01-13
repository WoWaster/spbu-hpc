#include <chrono>
#include <iomanip>
#include <iostream>
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

void compute_CE(int n, std::vector<double> &C, std::vector<double> &CE) {
  // single column of C*E, where E is a vector of ones
  // thus CE is a vector of row sums of C
  for (int i = 0; i < n; i++) {
    double tmp = 0.0;
    for (int k = 0; k < n; k++) {
      tmp += C[i * n + k];
    }
    CE[i] = tmp;
  }
}

double compute_trace(int n, std::vector<double> &B, std::vector<double> &CE) {
  double trace = 0.0;

  for (int i = 0; i < n; i++) {
    double tmp = 0.0;
    for (int k = 0; k < n; k++) {
      tmp += B[i * n + k] * CE[k];
    }
    trace += tmp;
  }

  return trace;
}

double compute_z1(int n, std::vector<double> &B, double Ex,
                  std::vector<double> &y) {
  double z1 = 0.0;

  for (int i = 0; i < n; i++) {
    double tmp = 0.0;
    for (int j = 0; j < n; j++) {
      tmp += B[i * n + j] * Ex * y[i];
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

void compute_A(int n, std::vector<double> &A, std::vector<double> &C,
               double trace, double z) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i * n + j] = trace * C[i * n + j] + z;
    }
  }

  for (int i = 0; i < n; i++) {
    A[i * n + i]++;
  }
}

void compute(int n, std::vector<double> &A, std::vector<double> &B,
             std::vector<double> &C, std::vector<double> &x,
             std::vector<double> &y, std::vector<double> &CE) {

  // Dot product of vector of ones (E) and x
  double Ex = std::accumulate(x.begin(), x.end(), 0.0);

  compute_CE(n, C, CE);

  double trace = compute_trace(n, B, CE);

  double z1 = compute_z1(n, B, Ex, y);

  double z2 = compute_z2(n, x, y);

  double z = z1 / z2;

  compute_A(n, A, C, trace, z);
}

int main(int argc, char *argv[]) {

  if (argc > 64)
    throw std::runtime_error("too many input parameters!");

  const std::vector<std::string> args(argv + 1, argv + argc);

  int n = -1;
  bool print_result = false;
  bool is_debug = false;

  // Not the best argument parser, I know
  for (const auto &arg : args) {
    if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--help] [--print] n" << std::endl;
      exit(EXIT_SUCCESS);
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
    n = std::stoi(arg);
  }

  re.seed(std::chrono::system_clock::now().time_since_epoch().count());

  std::vector<double> B(n * n);
  std::vector<double> C(n * n);
  std::vector<double> x(n);
  std::vector<double> y(n);
  std::vector<double> A(n * n);
  std::vector<double> CE(n);

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

  std::chrono::system_clock::time_point start_time =
      std::chrono::system_clock::now();

  compute(n, A, B, C, x, y, CE);

  std::chrono::milliseconds elapsed_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now() - start_time);

  if (print_result) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        std::cout << std::setprecision(10) << A[i * n + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << elapsed_milliseconds.count() << std::endl;

  return 0;
}
