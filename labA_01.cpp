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

std::vector<double> calculate(int n, std::vector<double> &B,
                              std::vector<double> &C, std::vector<double> &x,
                              std::vector<double> &y) {
  std::vector<double> A(n * n);

  // single column of C*E, where E is a vector of ones
  // thus CE is a vector of row sums of C
  std::vector<double> CE(n);

  double trace = 0.0;
  // Dot product of vector of ones (E) and x
  double Ex = std::accumulate(x.begin(), x.end(), 0.0);
  double z1 = 0.0;
  double z2 = 0.0;

#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < n; i++) {
      double tmp = 0.0;
      for (int k = 0; k < n; k++) {
        tmp += C[i * n + k];
      }
      CE[i] = tmp;
    }

#pragma omp for reduction(+ : trace)
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < n; k++) {
        trace += B[i * n + k] * CE[k];
      }
    }

#pragma omp for reduction(+ : z1)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        z1 += B[i * n + j] * Ex * y[i];
      }
    }
#pragma omp for reduction(+ : z2)
    for (int i = 0; i < n; i++) {
      z2 += x[i] * y[i];
    }

#pragma omp for collapse(2)
    for (int i = 0; i < n; i++) {
      // #pragma omp for
      for (int j = 0; j < n; j++) {
        A[i * n + j] = trace * C[i * n + j] + (i == j ? 1 : 0) + z1 / z2;
      }
    }
  }

  return A;
}

int main(int argc, char *argv[]) {

  if (argc > 64)
    throw std::runtime_error("too many input parameters!");

  const std::vector<std::string> args(argv + 1, argv + argc);

  int n = -1;
  bool print_result = false;

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
    n = std::stoi(arg);
  }

  re.seed(std::chrono::system_clock::now().time_since_epoch().count());

  std::vector<double> B(n * n);
  std::vector<double> C(n * n);
  std::vector<double> x(n);
  std::vector<double> y(n);

  generate(&B, random_e);
  generate(&C, random_e);
  generate(&x, random_e);
  generate(&y, random_e);

  std::cerr << "initialized values" << std::endl;

  std::chrono::system_clock::time_point start_time =
      std::chrono::system_clock::now();

  std::vector<double> A = calculate(n, B, C, x, y);

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

  std::cerr << "elapsed " << elapsed_milliseconds.count() << "ms" << std::endl;

  return 0;
}
