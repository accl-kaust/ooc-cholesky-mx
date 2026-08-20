/**
 *
 * @file example_dpotrf.c
 *
 *  PLASMA testing routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @brief Example of Cholesky factorization
 *
 * @version 2.8.0
 * @author Bilel Hadri
 * @date 2010-11-15
 * @generated d Fri Apr  1 11:03:07 2016
 *
 **/
#include <Eigen/Eigen>

#include <cblas.h>
#include <common.h>
#include <context.h>
#include <math.h>
#include <plasma.h>
#include <plasma_d_mixed.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#undef max
#undef min
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <vector>
#ifdef PLASMA_WITH_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
#include <core_blas.h>
#ifdef PLASMA_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

int check_factorization(int, double *, double *, int, int);

#ifdef PLASMA_WITH_CUDA
// Probe-based estimate of ||A - L L^T|| / ||A|| using random Gaussian probes.
// Matches chol_bench: n_probes=8, seed=0xdeadbeef, max_t err_t/||A x_t||,
// FP64 cuBLAS (Dgemv + 2 Dtrmv + Dnrm2/Daxpy). A column-major symmetric,
// L column-major lower-triangular (upper triangle assumed zero).
static double compute_relative_residual_probe(cublasHandle_t handle,
                                              const double *hA,
                                              const double *hL, int N,
                                              int n_probes, unsigned seed) {
  const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);
  double *dA = nullptr, *dL = nullptr;
  double *dx = nullptr, *dAx = nullptr, *dLLtx = nullptr;
  if (cudaMalloc(reinterpret_cast<void **>(&dA), bytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&dL), bytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&dx), N * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&dAx), N * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&dLLtx), N * sizeof(double)) != cudaSuccess) {
    if (dA) cudaFree(dA);
    if (dL) cudaFree(dL);
    if (dx) cudaFree(dx);
    if (dAx) cudaFree(dAx);
    if (dLLtx) cudaFree(dLLtx);
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dL, hL, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dA); cudaFree(dL); cudaFree(dx); cudaFree(dAx); cudaFree(dLLtx);
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> hx(static_cast<size_t>(N));
  const double one = 1.0, zero = 0.0, neg_one = -1.0;
  double max_rel = 0.0;

  for (int t = 0; t < n_probes; ++t) {
    for (int i = 0; i < N; ++i) hx[i] = dist(rng);
    if (cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) break;
    if (cudaMemcpy(dLLtx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) break;

    // Ax = A * x   (A symmetric, stored as full column-major)
    cublasDgemv(handle, CUBLAS_OP_N, N, N, &one, dA, N, dx, 1, &zero, dAx, 1);
    // y <- L^T y
    cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, N, dL, N, dLLtx, 1);
    // y <- L y    (now dLLtx = L L^T x)
    cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, N, dL, N, dLLtx, 1);

    double ax_norm = 0.0, err_norm = 0.0;
    cublasDnrm2(handle, N, dAx, 1, &ax_norm);
    // dAx <- dAx - dLLtx
    cublasDaxpy(handle, N, &neg_one, dLLtx, 1, dAx, 1);
    cublasDnrm2(handle, N, dAx, 1, &err_norm);
    if (ax_norm > 0.0) max_rel = std::max(max_rel, err_norm / ax_norm);
  }

  cudaFree(dA); cudaFree(dL); cudaFree(dx); cudaFree(dAx); cudaFree(dLLtx);
  return max_rel;
}
#endif

int IONE = 1;
int ISEED[4] = {0, 0, 0, 1}; /* initial seed for dlarnv() */

static bool parse_int_arg(const std::string &name, const std::string &value,
                          int &out) {
  try {
    size_t pos = 0;
    int v = std::stoi(value, &pos);
    if (pos != value.size()) {
      std::cerr << "Invalid value for " << name << ": '" << value << "'\n";
      return false;
    }
    out = v;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Invalid value for " << name << ": '" << value << "'";
    std::cerr << " (" << e.what() << ")\n";
    return false;
  }
}

#ifdef PLASMA_WITH_CUDA
static bool use_gpu_error_calc() {
  if (const char *env = std::getenv("MX_ERROR_GPU")) {
    return env[0] == '1';
  }
  const char *vis = std::getenv("CUDA_VISIBLE_DEVICES");
  if (!vis || vis[0] == '\0' || (vis[0] == '-' && vis[1] == '1')) {
    return false;
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return false;
  }
  return count > 0;
}
#ifdef __CUDACC__
__global__ void row_sum_abs_kernel(const double *A, const double *B,
                                   double *row_sums, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double sum = 0.0;
  for (int j = 0; j < N; ++j) {
    const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * N;
    sum += fabs(A[idx] - B[idx]);
  }
  row_sums[i] = sum;
}
#endif
#endif

int main(int argc, char **argv) {
  int cores = 2;
  int N = 1024; // default; can be inferred from bin if omitted/mismatched
  int nb = 128;
  std::string bin_path;
  std::string format;
  std::string mx_mode;
  std::string fp32_bucket;
  std::string fp16_bucket;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      if (!parse_int_arg("--n", argv[++i], N)) return 2;
    } else if (arg == "--nb" && i + 1 < argc) {
      if (!parse_int_arg("--nb", argv[++i], nb)) return 2;
    } else if (arg == "--bin" && i + 1 < argc) {
      bin_path = argv[++i];
    } else if (arg == "--cores" && i + 1 < argc) {
      if (!parse_int_arg("--cores", argv[++i], cores)) return 2;
    } else if (arg == "--format" && i + 1 < argc) {
      format = argv[++i];
    } else if (arg == "--mx-mode" && i + 1 < argc) {
      mx_mode = argv[++i];
    } else if (arg == "--fp32-bucket" && i + 1 < argc) {
      fp32_bucket = argv[++i];
    } else if (arg == "--fp16-bucket" && i + 1 < argc) {
      fp16_bucket = argv[++i];
    }
  }

  if (!format.empty()) {
    setenv("MX_FORCE_FORMAT", format.c_str(), 1);
  }
  if (!mx_mode.empty()) {
    setenv("MX_MX_MODE", mx_mode.c_str(), 1);
  }
  if (!fp32_bucket.empty()) {
    setenv("MX_BUCKET_FP32", fp32_bucket.c_str(), 1);
  }
  if (!fp16_bucket.empty()) {
    setenv("MX_BUCKET_FP16", fp16_bucket.c_str(), 1);
  }

#ifdef _OPENMP
  if (cores > 0) {
    omp_set_num_threads(cores);
  }
  Eigen::setNbThreads(cores);
  Eigen::initParallel();
#endif

  if (bin_path.empty()) {
    std::cerr << "usage: example_dpotrf_gpu --bin <path> [--n N] [--nb NB] "
                 "[--cores C] [--mx-mode M] [--fp32-bucket F] [--fp16-bucket F]\n"
                 "  --bin: raw binary file, row-major NxN float64, no header,\n"
                 "         symmetric positive-definite (N inferred from file\n"
                 "         size if --n is omitted). See data/README.md.\n";
    return 2;
  }
  std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
  if (!in) {
    std::cerr << "cannot open input bin: " << bin_path << "\n";
    return 2;
  }
  const std::streamsize bytes = in.tellg();
  in.seekg(0, std::ios::beg);

  const size_t count = static_cast<size_t>(bytes) / sizeof(double);
  const size_t dim = static_cast<size_t>(std::llround(std::sqrt(count)));
  if (dim * dim != count) {
    throw std::runtime_error("bin file size is not a square matrix");
  }
  if (N <= 0 || static_cast<size_t>(N) * N != count) {
    N = static_cast<int>(dim);
  }

  std::vector<double> data(static_cast<size_t>(N) * N);
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(double)));
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
      A(data.data(), N, N);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A_ref = A;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L = A;

  PLASMA_Init(cores);

  plasma_context_t *plasma = plasma_context_self();
  plasma->autotuning_enabled = 0;
  plasma->nb = nb;  // tile size

  auto start = std::chrono::high_resolution_clock::now();

  // Call the tile-first path to avoid the column-major → tile transform in
  // PLASMA_dpotrf_gpu.
  const auto status =
      PLASMA_dpotrf_gpu_reuse_data_table_mixed_precision(PlasmaLower, N,
                                                         L.data(), N);
  if (status != 0) {
    printf("factorization failed in %s:%d\n", __FILE__, __LINE__);
  }
  auto duration = std::chrono::high_resolution_clock::now() - start;
  // print duration
  std::cout << "Runtime: " << duration.count() << " microseconds" << std::endl;

  L.triangularView<Eigen::StrictlyUpper>().setZero();
  double error = 0;
  A_ref = A_ref.selfadjointView<Eigen::Lower>();
  double a_inf = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max:a_inf) schedule(static)
#endif
  for (int i = 0; i < N; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < N; ++j) {
      const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * N;
      row_sum += std::fabs(A_ref.data()[idx]);
    }
    if (row_sum > a_inf) a_inf = row_sum;
  }
  bool gpu_error_done = false;
#ifdef PLASMA_WITH_CUDA
  if (use_gpu_error_calc()) {
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);
    double *dA = nullptr;
    double *dL = nullptr;
    double *dLLt = nullptr;
    double *dRow = nullptr;
    cublasHandle_t handle = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&dA), bytes) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void **>(&dL), bytes) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void **>(&dLLt), bytes) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void **>(&dRow), static_cast<size_t>(N) * sizeof(double)) == cudaSuccess) {
      if (cudaMemcpy(dA, A_ref.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
          cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS) {
        const double alpha = 1.0;
        const double beta = 0.0;
        if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        N, N, N,
                        &alpha, dL, N,
                        dL, N,
                        &beta, dLLt, N) == CUBLAS_STATUS_SUCCESS) {
#ifdef __CUDACC__
          const int block = 256;
          const int grid = (N + block - 1) / block;
          row_sum_abs_kernel<<<grid, block>>>(dA, dLLt, dRow, N);
          if (cudaDeviceSynchronize() == cudaSuccess) {
            std::vector<double> row_sums(static_cast<size_t>(N));
            if (cudaMemcpy(row_sums.data(), dRow,
                           static_cast<size_t>(N) * sizeof(double),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
              double max_err = 0.0;
              for (int i = 0; i < N; ++i) {
                if (row_sums[static_cast<size_t>(i)] > max_err) {
                  max_err = row_sums[static_cast<size_t>(i)];
                }
              }
              error = max_err;
              gpu_error_done = true;
            }
          }
#endif
        }
      }
    }
    if (handle) cublasDestroy(handle);
    if (dA) cudaFree(dA);
    if (dL) cudaFree(dL);
    if (dLLt) cudaFree(dLLt);
    if (dRow) cudaFree(dRow);
  }
#endif
  if (!gpu_error_done) {
    std::vector<double> LLt(static_cast<size_t>(N) * N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
               N, N, N,
               1.0, L.data(), N,
               L.data(), N,
               0.0, LLt.data(), N);

    double max_err = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max:max_err) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
      double row_sum = 0.0;
      for (int j = 0; j < N; ++j) {
        const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * N;
        row_sum += std::fabs(A_ref.data()[idx] - LLt[idx]);
      }
      if (row_sum > max_err) max_err = row_sum;
    }
    error = max_err;
  }
  const double relative_error =
      (a_inf > 0.0) ? (error / a_inf) : std::numeric_limits<double>::quiet_NaN();
  std::cout << "error: " << error << std::endl;
  std::cout << "relative_error: " << relative_error << std::endl;

  // Probe-based relative residual estimate (matches chol_bench's metric).
  double relative_residual = std::numeric_limits<double>::quiet_NaN();
#ifdef PLASMA_WITH_CUDA
  {
    int n_probes = 8;
    unsigned probe_seed = 0xdeadbeefu;
    if (const char *env = std::getenv("MX_PROBE_COUNT")) {
      try { n_probes = std::max(1, std::stoi(env)); } catch (...) {}
    }
    if (const char *env = std::getenv("MX_PROBE_SEED")) {
      try { probe_seed = static_cast<unsigned>(std::stoul(env)); } catch (...) {}
    }
    cublasHandle_t probe_handle = nullptr;
    if (cublasCreate(&probe_handle) == CUBLAS_STATUS_SUCCESS) {
      relative_residual = compute_relative_residual_probe(
          probe_handle, A_ref.data(), L.data(), N, n_probes, probe_seed);
      cublasDestroy(probe_handle);
    }
  }
#endif
  std::cout << "relative_residual: " << relative_residual << std::endl;

  PLASMA_Finalize();

  return EXIT_SUCCESS;
}
