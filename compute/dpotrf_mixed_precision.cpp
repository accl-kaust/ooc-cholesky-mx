/**
 *
 * @file dpotrf.c
 *
 *  PLASMA computational routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.8.0
 * @author Jakub Kurzak
 * @date 2010-11-15
 * @generated d Fri Apr  1 11:02:54 2016
 *
 **/
#ifdef PLASMA_WITH_CUDA
#if PLASMA_WITH_HPCASIA24
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>

#include "common.h"
#include "cuda_fp8.h"
#include "mixed_precision.h"
#include "plasma_d_mixed.h"
#include "mx_emulation.h"
#include "mx_apply.h"
#include "ladder_selection.h"

static std::string bucket_format(const char *env_name) {
  if (const char *env = getenv(env_name)) {
    std::string fmt = env;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return fmt;
  }
  return {};
}

static bool is_device_or_managed(const void *p) {
  if (!p) return false;
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) return false;
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

static const char *dtype_to_string(int dtype) {
  switch (dtype) {
    case CUDA_R_64F:
      return "fp64";
    case CUDA_R_32F:
      return "fp32";
    case CUDA_R_16F:
      return "fp16";
    case CUDA_R_16BF:
      return "bf16";
    case CUDA_R_8F_E4M3:
      return "fp8_e4m3";
    default:
      return "unknown";
  }
}

struct QuantWorkspace {
  float *d_in = nullptr;
  float *d_out = nullptr;
  float *d_max = nullptr;
  size_t capacity = 0;  // elements
  size_t max_capacity = 0;  // max-values elements
};

static QuantWorkspace g_quant_ws;
// External linkage: incremented from mx_apply.h's templated tile-quant
// functions too, not just this file.
size_t g_pinned_bytes = 0;
size_t g_pinned_tiles = 0;
static size_t g_quant_ws_bytes = 0;

static void ensureQuantWorkspace(size_t elems, size_t max_elems) {
  if (g_quant_ws.capacity >= elems && g_quant_ws.max_capacity >= max_elems) {
    return;
  }
  if (g_quant_ws.d_in) {
    cudaFree(g_quant_ws.d_in);
    cudaFree(g_quant_ws.d_out);
    cudaFree(g_quant_ws.d_max);
  }
  g_quant_ws.capacity = elems;
  g_quant_ws.max_capacity = max_elems;
  cudaMalloc(reinterpret_cast<void **>(&g_quant_ws.d_in), elems * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&g_quant_ws.d_out), elems * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&g_quant_ws.d_max), max_elems * sizeof(float));
  g_quant_ws_bytes = elems * sizeof(float) * 2 + max_elems * sizeof(float);
}
#define FMULS_POTRF_ULL(__n) (__n * __n / 2 * (__n / 3 + 1) + __n / 3)
#define FADDS_POTRF_ULL(__n) (__n * ((__n * __n - 1) / 6))
#define FLOPS_DPOTRF_ULL(__n) (FMULS_POTRF_ULL((__n)) + FADDS_POTRF_ULL((__n)))

/***************************************************************************/
/**
 *
 * @ingroup double
 *
 *  PLASMA_dpotrf - Computes the Cholesky factorization of a symmetric positive
 *definite (or Hermitian positive definite in the complex case) matrix A. The
 *factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo =
 *PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] N
 *          The order of the matrix A. N >= 0.
 *
 * @param[in,out] A
 *          On entry, the symmetric positive definite (or Hermitian) matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly
 *lower triangular part of A is not referenced. If UPLO = 'L', the leading
 *N-by-N lower triangular part of A contains the lower triangular part of the
 *matrix A, and the strictly upper triangular part of A is not referenced. On
 *exit, if return value = 0, the factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T.
 *
 * @param[in] LDA
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *          \retval >0 if i, the leading minor of order i of A is not positive
 *definite, so the factorization could not be completed, and the solution has
 *not been computed.
 *
 *******************************************************************************
 *
 * @sa PLASMA_dpotrf_Tile
 * @sa PLASMA_dpotrf_Tile_Async
 * @sa PLASMA_cpotrf
 * @sa PLASMA_dpotrf
 * @sa PLASMA_spotrf
 * @sa PLASMA_dpotrs
 *
 ******************************************************************************/
void printMixedPrecisionTiledArray(MixedPrecisionTiledArray *array);

void makeMixedPrecisionTiledArray(MixedPrecisionTiledArray *array,
                                  const int uplo, const double *data,
                                  const size_t m, const size_t n,
                                  const size_t ld, const size_t nb) {

  g_pinned_tiles = 0;
  g_pinned_bytes = 0;

    array->nb = nb;
    array->mb = nb;
    array->m = array->n = n;
    array->nt = ((long)n - 1) / nb + 1;
    array->tiles = new MixedPrecisionTile[array->nt * array->nt]();

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>
      mappedA{data,
          static_cast<Eigen::Index>(m),
          static_cast<Eigen::Index>(n),
          {static_cast<Eigen::Index>(ld), 1}};

    long double normA = mappedA.norm();
    array->uplo = uplo;

    // Real native FP8 hardware requires compute capability >= 9
    // (Hopper+). On anything older (e.g. this A100), FP8 tiles must be
    // emulated instead (mx_apply::quant_plain_fp_tile) -- see the low
    // tier of Baseline's dispatch below.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties_v2(&deviceProp, 0);
    const bool hasFp8 = deviceProp.major >= 9;

    // Assuming CUDA headers are included:
    if (uplo == PlasmaLower) {
      for (int row = 0; row < array->nt; ++row) {
        for (int col = 0; col <= row; ++col) {
          auto &tile = array->tiles[col + row * array->nt];
          size_t nbRow = row == array->nt - 1 ? array->m - row * nb : nb;
          size_t nbCol = col == array->nt - 1 ? array->n - col * nb : nb;
          tile.m = nbRow;
          tile.n = nbCol;
          tile.ld = nbCol;
          tile.layout = CblasRowMajor;       
          auto mappedBlock = mappedA.block(row * nb, col * nb, nbRow, nbCol);
          long double normTile = mappedBlock.norm();
          auto epsilonRatio = array->nt * normTile / normA;

          const long double sourceEpsilon = []() {
            const char *env = getenv("MX_SOURCE_EPSILON");
            if (!env || !env[0]) {
              return static_cast<long double>(
                  std::numeric_limits<float>::epsilon());
            }
            char *end = nullptr;
            const long double v = std::strtold(env, &end);
            if (end == env || *end != '\0' || !(v > 0.0L)) {
              std::cout << "[WARN] Invalid MX_SOURCE_EPSILON='" << env
                        << "', using default float epsilon." << std::endl;
              return static_cast<long double>(
                  std::numeric_limits<float>::epsilon());
            }
            return v;
          }();
          // Baseline's low tier is always plain FP8_E4M3 (apply_plain_fp_quant,
          // no MX block scale) -- Baseline isn't an MX mode, so it no longer
          // reads an MX-named env var to decide this; it's just how Baseline
          // is defined. MX_SELECTION_CRITERIA=bound bypasses this whole
          // heuristic anyway for the 3 ladder modes (see ladder_selection.h).
          constexpr bool allow_low = true;
          const std::string fp32_bucket = bucket_format("MX_BUCKET_FP32");
          const std::string fp16_bucket = bucket_format("MX_BUCKET_FP16");
          // fp8_e4m3 has no MX shared scale, so its unit roundoff needs no
          // scale-aware underflow term -- format_unit_roundoff("fp8_e4m3")
          // is the whole story.
          const long double lowCutoff =
              sourceEpsilon / format_unit_roundoff("fp8_e4m3");

          // Thin forwarders into mx_apply.h -- the real quantization logic
          // lives there, templated on the Eigen expression type of
          // mappedBlock. Kept as same-named local lambdas so every call
          // site below reads exactly as it did before the split. (Only
          // apply_plain_fp_quant, log_tile_target, and apply_bucket_target
          // are still used here -- baseline's low tier is fixed to plain
          // FP8_E4M3; the ladder modes call mx_apply:: directly from
          // ladder_selection.h instead of through these.)
          auto apply_plain_fp_quant = [&](const char *label, int ebits, int mbits,
                                          float max_norm) {
            mx_apply::quant_plain_fp_tile(tile, row, col, mappedBlock, label, ebits, mbits, max_norm);
          };
          auto log_tile_target = [&](const char *target) {
            mx_apply::log_tile_target(row, col, target);
          };
          auto apply_bucket_target = [&](const std::string &bucket,
                                         const char *fallback) {
            mx_apply::apply_bucket(bucket, fallback, tile, row, col, mappedBlock);
          };

          if (row == col ) {
            tile.dtype = CUDA_R_64F;
            cudaMallocHost(&tile.data, sizeof(uint64_t) * tile.m * tile.n);
            g_pinned_tiles++;
            g_pinned_bytes += sizeof(uint64_t) * tile.m * tile.n;
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>>{
              static_cast<double *>(tile.data),
              static_cast<Eigen::Index>(tile.m),
              static_cast<Eigen::Index>(tile.n)} = mappedBlock.cast<double>();
            log_tile_target("fp64");
            std::cout << "[TILE_DTYPE] (" << row << ", " << col << ") "
                      << dtype_to_string(tile.dtype) << std::endl;
            continue;
          }

          if (mx_ladder::selection_criteria_is_bound()) {
            // Ladder IEEE / Ladder MX / Ladder MX+MXFP16 in the paper's
            // naming -- see compute/ladder_selection.h.
            mx_ladder::run(tile, row, col, mappedBlock, sourceEpsilon, normA,
                           normTile, array->nt, fp32_bucket, fp16_bucket);
          } else if (epsilonRatio > sourceEpsilon / std::numeric_limits<float>::epsilon()) {
            log_tile_target("fp64");
            tile.dtype = CUDA_R_64F;
            cudaMallocHost(&tile.data, sizeof(uint64_t) * tile.m * tile.n);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>{
                static_cast<double *>(tile.data),
                static_cast<Eigen::Index>(tile.m),
                static_cast<Eigen::Index>(tile.n)} = mappedBlock.cast<double>();
          } else if (epsilonRatio > sourceEpsilon / std::numeric_limits<Eigen::half>::epsilon()) {
            std::cout << "---------------- Tile (" << row << ", " << col
                        << ") selected for CUDA_R_32F -------------------"
                        << std::endl;
            apply_bucket_target(fp32_bucket, "fp32");
          } else if (!allow_low ||
                     epsilonRatio > lowCutoff) {
            apply_bucket_target(fp16_bucket, "fp16");
          } else if (hasFp8) {
            // Native hardware FP8 (Hopper+) -- no emulation needed.
            log_tile_target("fp8_e4m3");
            mx_apply::store_native_fp8_e4m3_tile(tile, mappedBlock);
          } else {
            // No native FP8 on this GPU (e.g. this A100) -- emulate it.
            log_tile_target("fp8_e4m3");
            apply_plain_fp_quant("FP8_E4M3", 4, 3, 448.0f);
          }

          std::cout << "[TILE_DTYPE] (" << row << ", " << col << ") "
                    << dtype_to_string(tile.dtype) << std::endl;
        }
      }
    } else {
      std::cout << "Only Lower matrix is supported so far" << std::endl;
    }
}

void uncompressMixedPrecisionTiledArray(const MixedPrecisionTiledArray *array,
                                        double *data, const size_t ld) {
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>
      mappedA{data,
              static_cast<Eigen::Index>(array->m),
              static_cast<Eigen::Index>(array->n),
              {static_cast<Eigen::Index>(ld), 1}};
  if (array->uplo == PlasmaLower) {
    for (int row = 0; row < array->nt; ++row) {
      for (int col = 0; col <= row; ++col) {
        const auto &tile = array->tiles[col + row * array->nt];
        auto mappedBlock =
            mappedA.block(row * array->nb, col * array->nb, tile.m, tile.n);
        // If tile data is on device/managed, stage to host for the map.
        bool deviceTile = is_device_or_managed(tile.data);
        switch (tile.dtype) {
          case CUDA_R_64F: {
            const double *src = static_cast<const double *>(tile.data);
            std::vector<double> hostBuf;
            const double *hostPtr = src;
            if (deviceTile) {
              hostBuf.resize(tile.m * tile.n);
              cudaMemcpy(hostBuf.data(), src, tile.m * tile.n * sizeof(double),
                         cudaMemcpyDeviceToHost);
              hostPtr = hostBuf.data();
            }
            mappedBlock = Eigen::Map<const Eigen::Matrix<
                double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>{
                hostPtr, static_cast<Eigen::Index>(tile.m),
                static_cast<Eigen::Index>(tile.n)};
            break;
          }
          case CUDA_R_32F: {
            const float *src = static_cast<const float *>(tile.data);
            std::vector<float> hostBuf;
            const float *hostPtr = src;
            if (deviceTile) {
              hostBuf.resize(tile.m * tile.n);
              cudaMemcpy(hostBuf.data(), src, tile.m * tile.n * sizeof(float),
                         cudaMemcpyDeviceToHost);
              hostPtr = hostBuf.data();
            }
            mappedBlock =
                Eigen::Map<const Eigen::Matrix<
                    float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>{
                    hostPtr, static_cast<Eigen::Index>(tile.m),
                    static_cast<Eigen::Index>(tile.n)}
                    .cast<double>();
            break;
          }
          case CUDA_R_16F: {
            const Eigen::half *src = static_cast<const Eigen::half *>(tile.data);
            std::vector<Eigen::half> hostBuf;
            const Eigen::half *hostPtr = src;
            if (deviceTile) {
              hostBuf.resize(tile.m * tile.n);
              cudaMemcpy(hostBuf.data(), src, tile.m * tile.n * sizeof(Eigen::half),
                         cudaMemcpyDeviceToHost);
              hostPtr = hostBuf.data();
            }
            mappedBlock =
                Eigen::Map<
                    const Eigen::Matrix<Eigen::half, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>{
                    hostPtr, static_cast<Eigen::Index>(tile.m),
                    static_cast<Eigen::Index>(tile.n)}
                    .cast<double>();
            break;
          }
          case CUDA_R_16BF: {
            const Eigen::bfloat16 *src =
                static_cast<const Eigen::bfloat16 *>(tile.data);
            std::vector<Eigen::bfloat16> hostBuf;
            const Eigen::bfloat16 *hostPtr = src;
            if (deviceTile) {
              hostBuf.resize(tile.m * tile.n);
              cudaMemcpy(hostBuf.data(), src,
                         tile.m * tile.n * sizeof(Eigen::bfloat16),
                         cudaMemcpyDeviceToHost);
              hostPtr = hostBuf.data();
            }
            mappedBlock =
                Eigen::Map<
                    const Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>{
                    hostPtr, static_cast<Eigen::Index>(tile.m),
                    static_cast<Eigen::Index>(tile.n)}
                    .cast<double>();
            break;
          }
          case CUDA_R_8F_E4M3: {
            const __nv_fp8_e4m3 *src =
                static_cast<const __nv_fp8_e4m3 *>(tile.data);
            std::vector<__nv_fp8_e4m3> hostBuf;
            const __nv_fp8_e4m3 *hostPtr = src;
            if (deviceTile) {
              hostBuf.resize(tile.m * tile.n);
              cudaMemcpy(hostBuf.data(), src,
                         tile.m * tile.n * sizeof(__nv_fp8_e4m3),
                         cudaMemcpyDeviceToHost);
              hostPtr = hostBuf.data();
            }
            mappedBlock =
                Eigen::Map<
                    const Eigen::Matrix<__nv_fp8_e4m3, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>{
                    hostPtr, static_cast<Eigen::Index>(tile.m),
                    static_cast<Eigen::Index>(tile.n)}
                    .cast<double>();
            break;
          }
          default:
            printf("Wrong datatype in %s:%d\n", __FILE__, __LINE__);
            break;
        }
      }
    }
  } else {
    printf("Only Lower matrix is supported so far\n");
  }
}

void printMixedPrecisionTiledArray(MixedPrecisionTiledArray *array) {
  if (array->uplo == PlasmaLower) {
    for (int row = 0; row < array->nt; ++row) {
      for (int col = 0; col <= row; ++col) {
        auto &tile = array->tiles[col + row * array->nt];
        switch (tile.dtype) {
          case CUDA_R_64F:
            printf("\033[31mFP64\033[0m    ");
            break;
          case CUDA_R_32F:
            printf("\033[33mFP32\033[0m    ");
            break;
          case CUDA_R_16F:
            printf("\033[34mFP16\033[0m    ");
            break;
          case CUDA_R_16BF:
            printf("\033[34mBF16\033[0m    ");
            break;
          case CUDA_R_8F_E4M3:
            printf("\033[32mFP8\033[0m     ");
            break;
          default:
            printf("Not supported data type\n");
        }
      }
      printf("\n");
    }
  } else {
    printf("Only Lower matrix is supported so far\n");
  }
}

void freeMixedPrecisionTiledArray(MixedPrecisionTiledArray *array) {
  if (array->uplo == PlasmaLower) {
    for (int row = 0; row < array->nt; ++row) {
      for (int col = 0; col <= row; ++col) {
        auto &tile = array->tiles[col + row * array->nt];
        if (!tile.data) continue;
        if (is_device_or_managed(tile.data)) {
          CHECK_CUDA(cudaFree(tile.data));
        } else {
          CHECK_CUDA(cudaFreeHost(tile.data));
        }
      }
    }
  } else {
    printf("Only Lower matrix is supported so far\n");
  }

  delete[] array->tiles;

  if (g_quant_ws.capacity) {
    cudaFree(g_quant_ws.d_in);
    cudaFree(g_quant_ws.d_out);
    cudaFree(g_quant_ws.d_max);
    g_quant_ws = {};
  }
}

// Real logic lives in mx_apply.h (mx_apply::requantize_tile_host) alongside
// quant_mx_tile/quant_plain_fp_tile -- kept as a thin forwarder here since
// control/mixed_precision.h declares this exact global name, callable from
// pdpotrf_gpu_mixed_precision.cpp.
void requantizeTileHost(const MixedPrecisionTile *tile) {
  mx_apply::requantize_tile_host(tile);
}

size_t getSizeofTileElement(int dtype) {
  switch (dtype) {
    case CUDA_R_64F:
      return sizeof(uint64_t);
    case CUDA_R_32F:
      return sizeof(uint32_t);
    case CUDA_R_16F:
      return sizeof(uint16_t);
    case CUDA_R_16BF:
      return sizeof(uint16_t);
    case CUDA_R_8F_E4M3:
      return sizeof(uint8_t);
    default:
      printf("Not supported data type\n");
      return 0;
  }
}

int getComputeType(int dtype) {
  switch (dtype) {
    case CUDA_R_64F:
      return CUBLAS_COMPUTE_64F;
    case CUDA_R_32F:
      return CUBLAS_COMPUTE_32F;
    case CUDA_R_16F:
      return CUBLAS_COMPUTE_32F;
    case CUDA_R_16BF:
      return CUBLAS_COMPUTE_32F;
    case CUDA_R_8F_E4M3:
      return CUBLAS_COMPUTE_32F;
    default:
      printf("Not supported data type\n");
      return 0;
  }
}

// int PLASMA_dpotrf_gpu_async_copy(PLASMA_enum uplo, int N,
//                                  double *A, int LDA)
//{
//   int NB;
//   int status;
//   plasma_context_t *plasma;
//   PLASMA_sequence *sequence = NULL;
//   PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
//   PLASMA_desc descA;
//
//   plasma = plasma_context_self();
//   if (plasma == NULL) {
//     plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
//     return PLASMA_ERR_NOT_INITIALIZED;
//   }
//   /* Check input arguments */
//   if (uplo != PlasmaUpper && uplo != PlasmaLower) {
//     plasma_error("PLASMA_dpotrf", "illegal value of uplo");
//     return -1;
//   }
//   if (N < 0) {
//     plasma_error("PLASMA_dpotrf", "illegal value of N");
//     return -2;
//   }
//   if (LDA < max(1, N)) {
//     plasma_error("PLASMA_dpotrf", "illegal value of LDA");
//     return -4;
//   }
//   /* Quick return */
//   if (max(N, 0) == 0)
//     return PLASMA_SUCCESS;
//
//   /* Tune NB depending on M, N & NRHS; Set NBNB */
//   status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
//   if (status != PLASMA_SUCCESS) {
//     plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
//     return status;
//   }
//
//   /* Set NT */
//   NB   = PLASMA_NB;
//
//   plasma_sequence_create(plasma, &sequence);
//
//   if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
//     plasma_dooplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N, sequence,
//     &request,
//                         plasma_desc_mat_free(&(descA)) );
//   } else {
//     plasma_diplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N,
//                        sequence, &request);
//   }
//
//   /* Call the tile interface */
//   PLASMA_dpotrf_gpu_async_copy_Tile_Async(uplo, &descA, sequence, &request);
//
//   if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
//     plasma_dooptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
//     plasma_dynamic_sync();
//     plasma_desc_mat_free(&descA);
//   } else {
//     plasma_diptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
//     plasma_dynamic_sync();
//   }
//
//   status = sequence->status;
//   plasma_sequence_destroy(plasma, sequence);
//
//   return status;
// }

int PLASMA_dpotrf_gpu_reuse_data_mixed_precision(PLASMA_enum uplo, int N,
                                                 double *A, int LDA) {
  int NB;
  int status;
  plasma_context_t *plasma;
  PLASMA_sequence *sequence = NULL;
  PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
  printf("Starting PLASMA_dpotrf_gpu_reuse_data_mixed_precision...\n");
  plasma = plasma_context_self();
  if (plasma == NULL) {
    plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
    return PLASMA_ERR_NOT_INITIALIZED;
  }
  /* Check input arguments */
  if (uplo != PlasmaUpper && uplo != PlasmaLower) {
    plasma_error("PLASMA_dpotrf", "illegal value of uplo");
    return -1;
  }
  if (N < 0) {
    plasma_error("PLASMA_dpotrf", "illegal value of N");
    return -2;
  }
  if (LDA < max(1, N)) {
    plasma_error("PLASMA_dpotrf", "illegal value of LDA");
    return -4;
  }
  /* Quick return */
  if (max(N, 0) == 0) return PLASMA_SUCCESS;

  /* Tune NB depending on M, N & NRHS; Set NBNB */
  status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
  if (status != PLASMA_SUCCESS) {
    plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
    return status;
  }

  /* Set NT */
  NB = (plasma->autotuning_enabled ? PLASMA_NB : plasma->nb);

  plasma_sequence_create(plasma, &sequence);

  MixedPrecisionTiledArray array;
  printf("Making mixed-precision tiled array...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  makeMixedPrecisionTiledArray(&array, uplo, A, N, N, LDA, NB);
  auto t1 = std::chrono::high_resolution_clock::now();

  //    if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
  //        plasma_dooplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N, sequence,
  //        &request,
  //                            plasma_desc_mat_free(&(descA)) );
  //    } else {
  //        plasma_diplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N,
  //                           sequence, &request);
  //    }

  auto start = std::chrono::high_resolution_clock::now();

  /* Call the tile interface */
  PLASMA_dpotrf_gpu_reuse_data_mixed_precision_Tile_Async(uplo, &array,
                                                          sequence, &request);

  auto duration = std::chrono::high_resolution_clock::now() - start;
  auto t2 = std::chrono::high_resolution_clock::now();
  const auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  const long double flops =
      (long double)(FLOPS_DPOTRF_ULL(static_cast<unsigned long long>(N)));
  const long double seconds = static_cast<long double>(micros) * 1e-6L;
  const long double tflops = flops / seconds / 1e12L;
  printf("FLOPS: %Lf\n", tflops);

  const auto tile_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  const auto factor_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - start)
          .count();
  const auto uncompress_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t2)
          .count();
  std::cout << "[TIMES] tile+quant(us): " << tile_us
            << ", factor(us): " << factor_us
            << ", uncompress(us): " << uncompress_us << std::endl;

  const long expected_tiles = (long)array.nt * (array.nt + 1) / 2;
  std::cout << "[STATS] nt: " << array.nt
            << ", expected lower tiles: " << expected_tiles
            << ", pinned tiles: " << g_pinned_tiles
            << ", pinned bytes: " << g_pinned_bytes
            << ", quant workspace bytes: " << g_quant_ws_bytes << std::endl;

  uncompressMixedPrecisionTiledArray(&array, A, LDA);
  freeMixedPrecisionTiledArray(&array);

  //    if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
  //        plasma_dooptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
  //        plasma_dynamic_sync();
  //        plasma_desc_mat_free(&descA);
  //    } else {
  //        plasma_diptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
  //        plasma_dynamic_sync();
  //    }

  status = sequence->status;
  plasma_sequence_destroy(plasma, sequence);

  return status;
}

int PLASMA_dpotrf_gpu_reuse_data_table_mixed_precision(PLASMA_enum uplo, int N,
                                                       double *A, int LDA) {
  int NB;
  int status;
  plasma_context_t *plasma;
  PLASMA_sequence *sequence = NULL;
  PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

  plasma = plasma_context_self();
  if (plasma == NULL) {
    plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
    return PLASMA_ERR_NOT_INITIALIZED;
  }
  /* Check input arguments */
  if (uplo != PlasmaUpper && uplo != PlasmaLower) {
    plasma_error("PLASMA_dpotrf", "illegal value of uplo");
    return -1;
  }
  if (N < 0) {
    plasma_error("PLASMA_dpotrf", "illegal value of N");
    return -2;
  }
  if (LDA < max(1, N)) {
    plasma_error("PLASMA_dpotrf", "illegal value of LDA");
    return -4;
  }
  /* Quick return */
  if (max(N, 0) == 0) return PLASMA_SUCCESS;

  /* Tune NB depending on M, N & NRHS; Set NBNB */
  status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
  if (status != PLASMA_SUCCESS) {
    plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
    return status;
  }

  /* Set NT */
  NB = (plasma->autotuning_enabled ? PLASMA_NB : plasma->nb);

  plasma_sequence_create(plasma, &sequence);

  MixedPrecisionTiledArray array;
  auto t0 = std::chrono::high_resolution_clock::now();
  makeMixedPrecisionTiledArray(&array, uplo, A, N, N, LDA, NB);
  auto t1 = std::chrono::high_resolution_clock::now();

  //    if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
  //        plasma_dooplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N, sequence,
  //        &request,
  //                            plasma_desc_mat_free(&(descA)) );
  //    } else {
  //        plasma_diplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N,
  //                           sequence, &request);
  //    }

  auto start = std::chrono::high_resolution_clock::now();

  /* Call the tile interface */
  PLASMA_dpotrf_gpu_reuse_data_table_mixed_precision_Tile_Async(
      uplo, &array, sequence, &request);

  auto duration = std::chrono::high_resolution_clock::now() - start;
  const auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  const long double flops =
      (long double)(FLOPS_DPOTRF_ULL(static_cast<unsigned long long>(N)));
  const long double seconds = static_cast<long double>(micros) * 1e-6L;
  const long double tflops = flops / seconds / 1e12L;
  printf("FLOPS: %Lf\n", tflops);

  auto t2 = std::chrono::high_resolution_clock::now();
  const auto tile_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  const auto factor_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - start)
          .count();
  const auto uncompress_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t2)
          .count();
  std::cout << "[TIMES] tile+quant(us): " << tile_us
            << ", factor(us): " << factor_us
            << ", uncompress(us): " << uncompress_us << std::endl;

  uncompressMixedPrecisionTiledArray(&array, A, LDA);
  freeMixedPrecisionTiledArray(&array);

  //    if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
  //        plasma_dooptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
  //        plasma_dynamic_sync();
  //        plasma_desc_mat_free(&descA);
  //    } else {
  //        plasma_diptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
  //        plasma_dynamic_sync();
  //    }

  status = sequence->status;
  plasma_sequence_destroy(plasma, sequence);

  return status;
}

int PLASMA_dpotrf_gpu_reuse_data_table_all_managed_mixed_precision(
    PLASMA_enum uplo, int N, double *A, int LDA) {
  int NB;
  int status;
  plasma_context_t *plasma;
  PLASMA_sequence *sequence = NULL;
  PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

  plasma = plasma_context_self();
  if (plasma == NULL) {
    plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
    return PLASMA_ERR_NOT_INITIALIZED;
  }
  /* Check input arguments */
  if (uplo != PlasmaUpper && uplo != PlasmaLower) {
    plasma_error("PLASMA_dpotrf", "illegal value of uplo");
    return -1;
  }
  if (N < 0) {
    plasma_error("PLASMA_dpotrf", "illegal value of N");
    return -2;
  }
  if (LDA < max(1, N)) {
    plasma_error("PLASMA_dpotrf", "illegal value of LDA");
    return -4;
  }
  /* Quick return */
  if (max(N, 0) == 0) return PLASMA_SUCCESS;

  /* Tune NB depending on M, N & NRHS; Set NBNB */
  status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
  if (status != PLASMA_SUCCESS) {
    plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
    return status;
  }

  /* Set NT */
  NB = (plasma->autotuning_enabled ? PLASMA_NB : plasma->nb);

  plasma_sequence_create(plasma, &sequence);

  MixedPrecisionTiledArray array;
  auto t0 = std::chrono::high_resolution_clock::now();
  makeMixedPrecisionTiledArray(&array, uplo, A, N, N, LDA, NB);
  auto t1 = std::chrono::high_resolution_clock::now();

  //    if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
  //        plasma_dooplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N, sequence,
  //        &request,
  //                            plasma_desc_mat_free(&(descA)) );
  //    } else {
  //        plasma_diplap2tile( descA, A, NB, NB, LDA, N, 0, 0, N, N,
  //                           sequence, &request);
  //    }

  auto start = std::chrono::high_resolution_clock::now();

  /* Call the tile interface */
  PLASMA_dpotrf_gpu_reuse_data_table_all_managed_mixed_precision_Tile_Async(
      uplo, &array, sequence, &request);

  auto duration = std::chrono::high_resolution_clock::now() - start;
  const auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  const long double flops =
      (long double)(FLOPS_DPOTRF_ULL(static_cast<unsigned long long>(N)));
  const long double seconds = static_cast<long double>(micros) * 1e-6L;
  const long double tflops = flops / seconds / 1e12L;
  printf("FLOPS: %Lf\n", tflops);

  auto t2 = std::chrono::high_resolution_clock::now();
  const auto tile_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  const auto factor_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - start)
          .count();
  const auto uncompress_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t2)
          .count();
  std::cout << "[TIMES] tile+quant(us): " << tile_us
            << ", factor(us): " << factor_us
            << ", uncompress(us): " << uncompress_us << std::endl;

  uncompressMixedPrecisionTiledArray(&array, A, LDA);
  freeMixedPrecisionTiledArray(&array);

  //    if ( PLASMA_TRANSLATION == PLASMA_OUTOFPLACE ) {
  //        plasma_dooptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
  //        plasma_dynamic_sync();
  //        plasma_desc_mat_free(&descA);
  //    } else {
  //        plasma_diptile2lap( descA, A, NB, NB, LDA, N,  sequence, &request);
  //        plasma_dynamic_sync();
  //    }

  status = sequence->status;
  plasma_sequence_destroy(plasma, sequence);

  return status;
}

/***************************************************************************/
/**
 *
 * @ingroup double_Tile
 *
 *  PLASMA_dpotrf_Tile - Computes the Cholesky factorization of a symmetric
 *positive definite or Hermitian positive definite matrix. Tile equivalent of
 *PLASMA_dpotrf(). Operates on matrices stored by tiles. All matrices are passed
 *through descriptors. All dimensions are taken from the descriptors.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] A
 *          On entry, the symmetric positive definite (or Hermitian) matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly
 *lower triangular part of A is not referenced. If UPLO = 'L', the leading
 *N-by-N lower triangular part of A contains the lower triangular part of the
 *matrix A, and the strictly upper triangular part of A is not referenced. On
 *exit, if return value = 0, the factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval >0 if i, the leading minor of order i of A is not positive
 *definite, so the factorization could not be completed, and the solution has
 *not been computed.
 *
 *******************************************************************************
 *
 * @sa PLASMA_dpotrf
 * @sa PLASMA_dpotrf_Tile_Async
 * @sa PLASMA_cpotrf_Tile
 * @sa PLASMA_dpotrf_Tile
 * @sa PLASMA_spotrf_Tile
 * @sa PLASMA_dpotrs_Tile
 *
 ******************************************************************************/
// int PLASMA_dpotrf_gpu_reuse_data_Tile(PLASMA_enum uplo, PLASMA_desc *A)
//{
//     plasma_context_t *plasma;
//     PLASMA_sequence *sequence = NULL;
//     PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
//     int status;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile", "PLASMA not initialized");
//         return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     plasma_sequence_create(plasma, &sequence);
//     PLASMA_dpotrf_gpu_reuse_data_Tile_Async(uplo, A, sequence, &request);
//     plasma_dynamic_sync();
//     status = sequence->status;
//     plasma_sequence_destroy(plasma, sequence);
//     return status;
// }
//
// int PLASMA_dpotrf_gpu_reuse_data_table_Tile(PLASMA_enum uplo, PLASMA_desc *A)
//{
//     plasma_context_t *plasma;
//     PLASMA_sequence *sequence = NULL;
//     PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
//     int status;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile", "PLASMA not initialized");
//         return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     plasma_sequence_create(plasma, &sequence);
//     PLASMA_dpotrf_gpu_reuse_data_table_Tile_Async(uplo, A, sequence,
//     &request); plasma_dynamic_sync(); status = sequence->status;
//     plasma_sequence_destroy(plasma, sequence);
//     return status;
// }
//
// int PLASMA_dpotrf_gpu_reuse_data_table_all_managed_Tile(PLASMA_enum uplo,
// PLASMA_desc *A)
//{
//     plasma_context_t *plasma;
//     PLASMA_sequence *sequence = NULL;
//     PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
//     int status;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile", "PLASMA not initialized");
//         return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     plasma_sequence_create(plasma, &sequence);
//     PLASMA_dpotrf_gpu_reuse_data_table_all_managed_Tile_Async(uplo, A,
//     sequence, &request); plasma_dynamic_sync(); status = sequence->status;
//     plasma_sequence_destroy(plasma, sequence);
//     return status;
// }
//
// int PLASMA_dpotrf_gpu_reuse_data_table_no_free_Tile(PLASMA_enum uplo,
// PLASMA_desc *A)
//{
//     plasma_context_t *plasma;
//     PLASMA_sequence *sequence = NULL;
//     PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
//     int status;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile", "PLASMA not initialized");
//         return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     plasma_sequence_create(plasma, &sequence);
//     PLASMA_dpotrf_gpu_reuse_data_table_no_free_Tile_Async(uplo, A, sequence,
//     &request); plasma_dynamic_sync(); status = sequence->status;
//     plasma_sequence_destroy(plasma, sequence);
//     return status;
// }
//
// int PLASMA_dpotrf_gpu_reuse_data_static_table_Tile(PLASMA_enum uplo,
// PLASMA_desc *A)
//{
//     plasma_context_t *plasma;
//     PLASMA_sequence *sequence = NULL;
//     PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
//     int status;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile", "PLASMA not initialized");
//         return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     plasma_sequence_create(plasma, &sequence);
//     PLASMA_dpotrf_gpu_reuse_data_static_table_Tile_Async(uplo, A, sequence,
//     &request); plasma_dynamic_sync(); status = sequence->status;
//     plasma_sequence_destroy(plasma, sequence);
//     return status;
// }

/***************************************************************************/
/**
 *
 * @ingroup double_Tile_Async
 *
 *  PLASMA_dpotrf_Tile_Async - Computes the Cholesky factorization of a
 *symmetric positive definite or Hermitian positive definite matrix.
 *Non-blocking equivalent of PLASMA_dpotrf_Tile(). May return before the
 *computation is finished. Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 *******************************************************************************
 *
 * @sa PLASMA_dpotrf
 * @sa PLASMA_dpotrf_Tile
 * @sa PLASMA_cpotrf_Tile_Async
 * @sa PLASMA_dpotrf_Tile_Async
 * @sa PLASMA_spotrf_Tile_Async
 * @sa PLASMA_dpotrs_Tile_Async
 *
 ******************************************************************************/
// int PLASMA_dpotrf_gpu_async_copy_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A,
//                                             PLASMA_sequence *sequence,
//                                             PLASMA_request *request)
//{
//     PLASMA_desc descA;
//     plasma_context_t *plasma;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not
//         initialized"); return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     if (sequence == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
//         return PLASMA_ERR_UNALLOCATED;
//     }
//     if (request == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
//         return PLASMA_ERR_UNALLOCATED;
//     }
//     /* Check sequence status */
//     if (sequence->status == PLASMA_SUCCESS)
//         request->status = PLASMA_SUCCESS;
//     else
//         return plasma_request_fail(sequence, request,
//         PLASMA_ERR_SEQUENCE_FLUSHED);
//
//     /* Check descriptors for correctness */
//     if (plasma_desc_check(A) != PLASMA_SUCCESS) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
//         return plasma_request_fail(sequence, request,
//         PLASMA_ERR_ILLEGAL_VALUE);
//     } else {
//         descA = *A;
//     }
//     /* Check input arguments */
//     if (descA.nb != descA.mb) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles
//         supported"); return plasma_request_fail(sequence, request,
//         PLASMA_ERR_ILLEGAL_VALUE);
//     }
//     if (uplo != PlasmaUpper && uplo != PlasmaLower) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
//         return plasma_request_fail(sequence, request, -1);
//     }
//     /* Quick return */
///*
//    if (max(N, 0) == 0)
//        return PLASMA_SUCCESS;
//*/
//    plasma_parallel_call_4(plasma_pdpotrf_gpu_async_copy,
//        PLASMA_enum, uplo,
//        PLASMA_desc, descA,
//        PLASMA_sequence*, sequence,
//        PLASMA_request*, request);
//
//    return PLASMA_SUCCESS;
//}

int PLASMA_dpotrf_gpu_reuse_data_mixed_precision_Tile_Async(
    PLASMA_enum uplo, MixedPrecisionTiledArray *A, PLASMA_sequence *sequence,
    PLASMA_request *request) {
  MixedPrecisionTiledArray descA;
  plasma_context_t *plasma;

  plasma = plasma_context_self();
  if (plasma == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not initialized");
    return PLASMA_ERR_NOT_INITIALIZED;
  }
  if (sequence == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
    return PLASMA_ERR_UNALLOCATED;
  }
  if (request == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
    return PLASMA_ERR_UNALLOCATED;
  }
  /* Check sequence status */
  if (sequence->status == PLASMA_SUCCESS)
    request->status = PLASMA_SUCCESS;
  else
    return plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);

  /* Check descriptors for correctness */
  if (PLASMA_SUCCESS /*plasma_desc_check(A)*/ != PLASMA_SUCCESS) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
    return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
  } else {
    descA = *A;
  }
  /* Check input arguments */
  if (descA.nb != descA.mb) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles supported");
    return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
  }
  if (uplo != PlasmaUpper && uplo != PlasmaLower) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
    return plasma_request_fail(sequence, request, -1);
  }
  /* Quick return */
  /*
      if (max(N, 0) == 0)
          return PLASMA_SUCCESS;
  */
  plasma_parallel_call_4(plasma_pdpotrf_gpu_reuse_data_mixed_precision,
                         PLASMA_enum, uplo, MixedPrecisionTiledArray, descA,
                         PLASMA_sequence *, sequence, PLASMA_request *,
                         request);

  return PLASMA_SUCCESS;
}

int PLASMA_dpotrf_gpu_reuse_data_table_mixed_precision_Tile_Async(
    PLASMA_enum uplo, MixedPrecisionTiledArray *A, PLASMA_sequence *sequence,
    PLASMA_request *request) {
  MixedPrecisionTiledArray descA;
  plasma_context_t *plasma;

  plasma = plasma_context_self();
  if (plasma == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not initialized");
    return PLASMA_ERR_NOT_INITIALIZED;
  }
  if (sequence == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
    return PLASMA_ERR_UNALLOCATED;
  }
  if (request == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
    return PLASMA_ERR_UNALLOCATED;
  }
  /* Check sequence status */
  if (sequence->status == PLASMA_SUCCESS)
    request->status = PLASMA_SUCCESS;
  else
    return plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);

  /* Check descriptors for correctness */
  if (PLASMA_SUCCESS /*plasma_desc_check(A)*/ != PLASMA_SUCCESS) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
    return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
  } else {
    descA = *A;
  }
  /* Check input arguments */
  if (descA.nb != descA.mb) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles supported");
    return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
  }
  if (uplo != PlasmaUpper && uplo != PlasmaLower) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
    return plasma_request_fail(sequence, request, -1);
  }
  /* Quick return */
  /*
      if (max(N, 0) == 0)
          return PLASMA_SUCCESS;
  */
  plasma_parallel_call_4(plasma_pdpotrf_gpu_reuse_data_table_mixed_precision,
                         PLASMA_enum, uplo, MixedPrecisionTiledArray, descA,
                         PLASMA_sequence *, sequence, PLASMA_request *,
                         request);

  return PLASMA_SUCCESS;
}

int PLASMA_dpotrf_gpu_reuse_data_table_all_managed_mixed_precision_Tile_Async(
    PLASMA_enum uplo, MixedPrecisionTiledArray *A, PLASMA_sequence *sequence,
    PLASMA_request *request) {
  MixedPrecisionTiledArray descA;
  plasma_context_t *plasma;

  plasma = plasma_context_self();
  if (plasma == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not initialized");
    return PLASMA_ERR_NOT_INITIALIZED;
  }
  if (sequence == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
    return PLASMA_ERR_UNALLOCATED;
  }
  if (request == NULL) {
    plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
    return PLASMA_ERR_UNALLOCATED;
  }
  /* Check sequence status */
  if (sequence->status == PLASMA_SUCCESS)
    request->status = PLASMA_SUCCESS;
  else
    return plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);

  /* Check descriptors for correctness */
  if (PLASMA_SUCCESS /*plasma_desc_check(A)*/ != PLASMA_SUCCESS) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
    return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
  } else {
    descA = *A;
  }
  /* Check input arguments */
  if (descA.nb != descA.mb) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles supported");
    return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
  }
  if (uplo != PlasmaUpper && uplo != PlasmaLower) {
    plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
    return plasma_request_fail(sequence, request, -1);
  }
  /* Quick return */
  /*
      if (max(N, 0) == 0)
          return PLASMA_SUCCESS;
  */
  plasma_parallel_call_4(
      plasma_pdpotrf_gpu_reuse_data_table_all_managed_mixed_precision,
      PLASMA_enum, uplo, MixedPrecisionTiledArray, descA, PLASMA_sequence *,
      sequence, PLASMA_request *, request);

  return PLASMA_SUCCESS;
}

// int PLASMA_dpotrf_gpu_reuse_data_table_no_free_Tile_Async(PLASMA_enum uplo,
// PLASMA_desc *A,
//                                                           PLASMA_sequence
//                                                           *sequence,
//                                                           PLASMA_request
//                                                           *request)
//{
//     PLASMA_desc descA;
//     plasma_context_t *plasma;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not
//         initialized"); return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     if (sequence == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
//         return PLASMA_ERR_UNALLOCATED;
//     }
//     if (request == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
//         return PLASMA_ERR_UNALLOCATED;
//     }
//     /* Check sequence status */
//     if (sequence->status == PLASMA_SUCCESS)
//         request->status = PLASMA_SUCCESS;
//     else
//         return plasma_request_fail(sequence, request,
//         PLASMA_ERR_SEQUENCE_FLUSHED);
//
//     /* Check descriptors for correctness */
//     if (plasma_desc_check(A) != PLASMA_SUCCESS) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
//         return plasma_request_fail(sequence, request,
//         PLASMA_ERR_ILLEGAL_VALUE);
//     } else {
//         descA = *A;
//     }
//     /* Check input arguments */
//     if (descA.nb != descA.mb) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles
//         supported"); return plasma_request_fail(sequence, request,
//         PLASMA_ERR_ILLEGAL_VALUE);
//     }
//     if (uplo != PlasmaUpper && uplo != PlasmaLower) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
//         return plasma_request_fail(sequence, request, -1);
//     }
//     /* Quick return */
//     /*
//         if (max(N, 0) == 0)
//             return PLASMA_SUCCESS;
//     */
//     plasma_parallel_call_4(plasma_pdpotrf_gpu_reuse_data_table_no_free,
//                            PLASMA_enum, uplo,
//                            PLASMA_desc, descA,
//                            PLASMA_sequence*, sequence,
//                            PLASMA_request*, request);
//
//     return PLASMA_SUCCESS;
// }
//
// int PLASMA_dpotrf_gpu_reuse_data_static_table_Tile_Async(PLASMA_enum uplo,
// PLASMA_desc *A,
//                                                               PLASMA_sequence
//                                                               *sequence,
//                                                               PLASMA_request
//                                                               *request)
//{
//     PLASMA_desc descA;
//     plasma_context_t *plasma;
//
//     plasma = plasma_context_self();
//     if (plasma == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not
//         initialized"); return PLASMA_ERR_NOT_INITIALIZED;
//     }
//     if (sequence == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
//         return PLASMA_ERR_UNALLOCATED;
//     }
//     if (request == NULL) {
//         plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
//         return PLASMA_ERR_UNALLOCATED;
//     }
//     /* Check sequence status */
//     if (sequence->status == PLASMA_SUCCESS)
//         request->status = PLASMA_SUCCESS;
//     else
//         return plasma_request_fail(sequence, request,
//         PLASMA_ERR_SEQUENCE_FLUSHED);
//
//     /* Check descriptors for correctness */
//     if (plasma_desc_check(A) != PLASMA_SUCCESS) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
//         return plasma_request_fail(sequence, request,
//         PLASMA_ERR_ILLEGAL_VALUE);
//     } else {
//         descA = *A;
//     }
//     /* Check input arguments */
//     if (descA.nb != descA.mb) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles
//         supported"); return plasma_request_fail(sequence, request,
//         PLASMA_ERR_ILLEGAL_VALUE);
//     }
//     if (uplo != PlasmaUpper && uplo != PlasmaLower) {
//         plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
//         return plasma_request_fail(sequence, request, -1);
//     }
//     /* Quick return */
//     /*
//         if (max(N, 0) == 0)
//             return PLASMA_SUCCESS;
//     */
//     plasma_parallel_call_4(plasma_pdpotrf_gpu_reuse_data_static_table,
//                            PLASMA_enum, uplo,
//                            PLASMA_desc, descA,
//                            PLASMA_sequence*, sequence,
//                            PLASMA_request*, request);
//
//     return PLASMA_SUCCESS;
// }
#endif
#endif