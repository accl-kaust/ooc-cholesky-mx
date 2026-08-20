#include <cuda_runtime.h>
/**
 *
 * @file pdpotrf.c
 *
 *  PLASMA auxiliary routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.8.0
 * @author Jakub Kurzak
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @generated d Fri Apr  1 11:02:57 2016
 *
 **/
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "dist.h"
#include "dist_tile_copy.h"
#define A(m, n) BLKADDR(A, double, m, n)
#define A_AFFINITY(m, n) BLKADDR_AFFINITY(A, double, m, n)
/***************************************************************************/
/**
 *  Parallel tile Cholesky factorization - static scheduling
 **/

namespace {
void cudaDeleter(double *p) { CHECK_CUDA(cudaFree(p)); }
void *cudaAllocator(size_t size) {
  void *p;
  CHECK_CUDA(cudaMalloc(&p, size));
  return p;
}
}  // namespace

// struct DataTable {
//   union Coord {
//     std::pair<int, int> pair;
//     long dummy;
//   };
//   static_assert(sizeof(Coord) == 2 * sizeof(int),
//                 "We assume long is doubled int, please report this bug");
//   std::mutex mutex{};
//   using Map =
//       std::unordered_map<long, std::tuple<std::shared_ptr<double>, bool>>;
//   Map map{};
//   std::shared_ptr<double> diagonal;
//   int diagIdx;
//   enum { SHARED_PTR = 0, FLAG = 1 };
// };
//
//  std::tuple<std::shared_ptr<double>, bool, bool *> lookupTable(
//      DataTable::Map &map, int mapBound, std::pair<int, int> pair, size_t
//      size)
//      {
//    DataTable::Coord coord{.pair = pair};
//    std::shared_ptr<double> buffer{nullptr};
//    auto find = map.find(coord.dummy);
//    bool exist = false;
//    if (find != map.end()) {
//      buffer = std::get<DataTable::SHARED_PTR>(find->second);
//      exist = true;
//    } else {
//      if (map.size() >= mapBound) {
//        do {
//          for (auto &[key, value] : map) {
//            if (std::get<DataTable::SHARED_PTR>(value).use_count() == 1) {
//              buffer = std::get<DataTable::SHARED_PTR>(value);
//              map.insert(
//                  {coord.dummy,
//                   {std::move(std::get<DataTable::SHARED_PTR>(value)),
//                   false}});
//              find = map.find(coord.dummy);
//              map.erase(key);
//              break;
//            }
//          }
//          // plasma_yield();
//        } while (buffer == nullptr);
//      } else {
//        buffer.reset((double *)cudaAllocator(size), cudaDeleter);
//        map.insert({coord.dummy, {buffer, false}});
//        find = map.find(coord.dummy);
//      }
//    }
//    return {std::move(buffer), exist,
//    &std::get<DataTable::FLAG>(find->second)};
//  }
//  }  // namespace

// deduceMemcpyKind moved verbatim to compute/dist_tile_copy.h (#included
// above) -- unchanged logic, just relocated.
void plasma_pdpotrf_gpu_dist(plasma_context_t *plasma) {
  const int rank = PLASMA_RANK;
  const int device = plasma->rankMapping[rank];
  double start, end;
  cudaSetDevice(device);

  if (rank == 0) {
    plasma->volumeCPU2GPU = 0;
    plasma->volumeGPU2CPU = 0;
    //    aux = plasma->aux;
    //    plasma->aux = (void *)new DataTable;
  }
  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  if (uplo == PlasmaUpper) {
    printf("Upper is not supported yet\n");
    return;
  }
  if (sequence->status != PLASMA_SUCCESS) return;
  ss_init(A.nt, A.nt, 0);
  // synced in ss_init
//  auto dataTable = (DataTable *)plasma->aux;
#define volumeCPU2GPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeCPU2GPU)> &>( \
      plasma->volumeCPU2GPU))
#define volumeGPU2CPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeGPU2CPU)> &>( \
      plasma->volumeGPU2CPU))

  k = 0;
  m = PLASMA_RANK;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;
  int kDiag = -1;

  size_t workspaceInBytes;
  CHECK_CUSOLVER(cusolverDnPotrf_bufferSize(
      plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
      CUBLAS_FILL_MODE_LOWER, A.nb, CUDA_R_64F, NULL, A.nb, CUDA_R_64F,
      &workspaceInBytes));
  const size_t bufferSizeInByte = (unsigned long)(A.nb) * A.nb * sizeof(double);

  double *diagonal, *localDestination, *buffer, *bufferA, *bufferB;
  CHECK_CUDA(
      cudaMalloc(reinterpret_cast<void **>(&buffer), 4 * bufferSizeInByte));
  bufferA = buffer;
  bufferB = bufferA + (unsigned long)(A.nb) * A.nb;
  diagonal = bufferB + (unsigned long)(A.nb) * A.nb;
  localDestination = diagonal + (unsigned long)(A.nb) * A.nb;
  void *pBuffer;
  CHECK_CUDA(
      cudaMalloc(&pBuffer, workspaceInBytes +
                               (sizeof(int) - workspaceInBytes % sizeof(int)) +
                               sizeof(int)));
  infoDevice = (int *)((long)pBuffer + workspaceInBytes +
                       (sizeof(int) - workspaceInBytes % sizeof(int)));
  double one = 1., none = -1.;

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
            auto kind = deduceMemcpyKind(localDestination, A(k, k));
            CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn, kind,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUSOLVER(cusolverDnPotrf(
              plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
              CUBLAS_FILL_MODE_LOWER, tempkn, CUDA_R_64F, localDestination,
              ldak, CUDA_R_64F, pBuffer, workspaceInBytes, infoDevice));
          //                    CORE_dpotrf(
          //                        PlasmaLower,
          //                        tempkn,
          //                        A(k, k), ldak,
          //                        &info);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dpotrf(PlasmaUpper, tempkn, A(k, k), ldak, &info);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        auto kindOut = deduceMemcpyKind(A(k, k), localDestination);
        CHECK_CUDA(cudaMemcpy2DAsync(
          A(k, k), ldak * sizeof(double), localDestination,
          ldak * sizeof(double), tempkn * sizeof(double), tempkn, kindOut,
          plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldak * sizeof(double) * tempkn;
        CHECK_CUDA(cudaMemcpyAsync(&info, infoDevice, sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   plasma->cuda_stream[rank]));
        volumeGPU2CPU += sizeof(int);
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        if (info != 0) {
          plasma_request_fail(sequence, request, info + A.nb * k);
          ss_abort();
        }
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(k, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
            auto kind = deduceMemcpyKind(localDestination, A(k, k));
            CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn, kind,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        start = get_current_time();
        auto kindA = deduceMemcpyKind(bufferA, A(k, n));
        CHECK_CUDA(cudaMemcpy2DAsync(
          bufferA, ldak * sizeof(double), A(k, n), ldak * sizeof(double),
          tempkn * sizeof(double), A.nb, kindA, plasma->cuda_stream[rank]));
        volumeCPU2GPU += ldak * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_C2G, start, end);
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDsyrk(plasma->cublas_handle[rank],
                                   CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tempkn,
                                   A.nb, &none, bufferA, ldak, &one,
                                   localDestination, ldak));
          //                    CORE_dsyrk(
          //                         PlasmaLower, PlasmaNoTrans,
          //                         tempkn, A.nb,
          //                         -1.0, A(k, n), ldak,
          //                          1.0, A(k, k), ldak);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dsyrk(PlasmaUpper, PlasmaTrans, tempkn, A.nb, -1.0, A(n, k),
                     ldan, 1.0, A(k, k), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    } else {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
            auto kind = deduceMemcpyKind(localDestination, A(m, k));
            CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kind,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, k, 1);

        if (kDiag != k) {
          start = get_current_time();
            auto kindDiag = deduceMemcpyKind(diagonal, A(k, k));
            CHECK_CUDA(cudaMemcpy2DAsync(
              diagonal, ldak * sizeof(double), A(k, k), ldak * sizeof(double),
              A.nb * sizeof(double), A.nb, kindDiag, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          kDiag = k;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDtrsm(
              plasma->cublas_handle[rank], CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tempmn,
              A.nb, &zone, diagonal, ldak, localDestination, ldam));
          //                    CORE_dtrsm(
          //                        PlasmaRight, PlasmaLower, PlasmaTrans,
          //                        PlasmaNonUnit, tempmn, A.nb, zone, A(k, k),
          //                        ldak,
          //                              A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit, A.nb,
                     tempmn, zone, A(k, k), ldak, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        auto kindBack = deduceMemcpyKind(A(m, k), localDestination);
        CHECK_CUDA(cudaMemcpy2DAsync(
          A(m, k), ldam * sizeof(double), localDestination,
          ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindBack,
          plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldam * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(m, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination, A(m, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindLocal,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        start = get_current_time();
        auto kindB = deduceMemcpyKind(bufferB, A(k, n));
        CHECK_CUDA(cudaMemcpy2DAsync(
          bufferB, ldak * sizeof(double), A(k, n), ldak * sizeof(double),
          A.nb * sizeof(double), A.nb, kindB, plasma->cuda_stream[rank]));
        volumeCPU2GPU += ldak * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_C2G, start, end);

        ss_cond_wait(m, n, 1);
        start = get_current_time();
        auto kindA2 = deduceMemcpyKind(bufferA, A(m, n));
        CHECK_CUDA(cudaMemcpy2DAsync(
          bufferA, ldam * sizeof(double), A(m, n), ldam * sizeof(double),
          tempmn * sizeof(double), A.nb, kindA2,
          plasma->cuda_stream[rank]));
        volumeCPU2GPU += ldam * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_C2G, start, end);
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDgemm(plasma->cublas_handle[rank], CUBLAS_OP_N,
                                   CUBLAS_OP_T, tempmn, A.nb, A.nb, &mzone,
                                   bufferA, ldam, bufferB, ldak, &zone,
                                   localDestination, ldam));
          //                    CORE_dgemm(
          //                        PlasmaNoTrans, PlasmaTrans,
          //                        tempmn, A.nb, A.nb,
          //                        mzone, A(m, n), ldam,
          //                               A(k, n), ldak,
          //                         zone, A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dgemm(PlasmaTrans, PlasmaNoTrans, A.nb, tempmn, A.nb, mzone,
                     A(n, k), ldan, A(n, m), ldan, zone, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  CHECK_CUDA(cudaFree(pBuffer));
  CHECK_CUDA(cudaFree(buffer));
  ss_finalize();
//  if (rank == 0) {
//    plasma->aux = aux;
//    delete dataTable;
//  }
#undef volumeCPU2GPU
#undef volumeGPU2CPU
}

namespace {
struct DataTable {
  union Coord {
    std::pair<int, int> pair;
    long dummy;
  };
  static_assert(sizeof(Coord) == 2 * sizeof(int),
                "We assume long is doubled int, please report this bug");
  std::mutex mutex{};
  using MapValue = std::tuple<std::shared_ptr<double>, bool>;
  using Map = std::unordered_map<long, MapValue>;
  Map map{};
  std::shared_ptr<double> diagonal;
  int diagIdx{-1};
  int mapBound{-1};
  enum { SHARED_PTR = 0, FLAG = 1 };
  bool canAllocateNew{true};
};

std::tuple<std::shared_ptr<double>, bool, bool *> lookupTable(
    DataTable::Map &map, int mapBound, std::pair<int, int> pair, size_t size,
    bool canAllocateNew = true) {
  DataTable::Coord coord{.pair = pair};
  std::shared_ptr<double> buffer{nullptr};
  auto find = map.find(coord.dummy);
  bool exist = false;
  if (find != map.end()) {
    buffer = std::get<DataTable::SHARED_PTR>(find->second);
    exist = true;
  } else {
    if (!canAllocateNew || map.size() >= mapBound) {
      do {
        for (auto &[key, value] : map) {
          if (std::get<DataTable::SHARED_PTR>(value).use_count() == 1) {
            buffer = std::get<DataTable::SHARED_PTR>(value);
            map.insert(
                {coord.dummy,
                 {std::move(std::get<DataTable::SHARED_PTR>(value)), false}});
            find = map.find(coord.dummy);
            map.erase(key);
            break;
          }
        }
        // plasma_yield();
      } while (buffer == nullptr);
    } else {
      buffer.reset((double *)cudaAllocator(size), cudaDeleter);
      map.insert({coord.dummy, {buffer, false}});
      find = map.find(coord.dummy);
    }
  }
  return {std::move(buffer), exist, &std::get<DataTable::FLAG>(find->second)};
}

void setLandedTable(DataTable::Map &map, std::pair<int, int> pair, bool flag) {
  DataTable::Coord coord{.pair = pair};
  auto find = map.find(coord.dummy);
  if (find != map.end()) {
    std::get<DataTable::FLAG>(find->second) = flag;
  } else {
    printf("Unexpected situation happened at %s:%d, please report the bug\n",
           __FILE__, __LINE__);
  }
}

int getMapBound(int nb) {
  size_t freeSize, totalSize;
  cudaMemGetInfo(&freeSize, &totalSize);
  // NOTE: Jie: Use 90% free memory
  return freeSize / sizeof(double) * 9 / nb / nb / 10;
}
}  // namespace

void plasma_pdpotrf_gpu_reuse_data_dist(plasma_context_t *plasma) {
#undef A
#define A(m, n) A_AFFINITY(m, n)
  const int rank = PLASMA_RANK;
  double start, end;
  const int device = plasma->rankMapping[rank];
  cudaSetDevice(device);
  if (rank == 0) {
    plasma->volumeCPU2GPU = 0;
    plasma->volumeGPU2CPU = 0;
  }
  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  if (uplo == PlasmaUpper) {
    printf("Upper is not supported yet\n");
    return;
  }
  if (sequence->status != PLASMA_SUCCESS) return;
  ss_init(A.nt, A.nt, 0);
  // synced in ss_init
#define volumeCPU2GPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeCPU2GPU)> &>( \
      plasma->volumeCPU2GPU))
#define volumeGPU2CPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeGPU2CPU)> &>( \
      plasma->volumeGPU2CPU))

  k = 0;
  m = PLASMA_RANK;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;
  int kDiag = -1;

  size_t workspaceInBytes;
  CHECK_CUSOLVER(cusolverDnPotrf_bufferSize(
      plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
      CUBLAS_FILL_MODE_LOWER, A.nb, CUDA_R_64F, NULL, A.nb, CUDA_R_64F,
      &workspaceInBytes));
  double *buffer, *bufferA, *bufferB, *diagonal, *localDestination;
  const size_t bufferSizeInByte = (unsigned long)(A.nb) * A.nb * sizeof(double);
  cudaMalloc((void **)&buffer, 4 * bufferSizeInByte);
  bufferA = buffer;
  bufferB = bufferA + (unsigned long)(A.nb) * A.nb;
  diagonal = bufferB + (unsigned long)(A.nb) * A.nb;
  localDestination = diagonal + (unsigned long)(A.nb) * A.nb;
  void *pBuffer;
  CHECK_CUDA(
      cudaMalloc(&pBuffer, workspaceInBytes +
                               (sizeof(int) - workspaceInBytes % sizeof(int)) +
                               sizeof(int)));
  infoDevice = (int *)((long)pBuffer + workspaceInBytes +
                       (sizeof(int) - workspaceInBytes % sizeof(int)));
  double one = 1., none = -1.;

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination, A(k, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              kindLocal, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUSOLVER(cusolverDnPotrf(
              plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
              CUBLAS_FILL_MODE_LOWER, tempkn, CUDA_R_64F, localDestination,
              ldak, CUDA_R_64F, pBuffer, workspaceInBytes, infoDevice));
          //                    CORE_dpotrf(
          //                        PlasmaLower,
          //                        tempkn,
          //                        A(k, k), ldak,
          //                        &info);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dpotrf(PlasmaUpper, tempkn, A(k, k), ldak, &info);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        auto kindOut = deduceMemcpyKind(A(k, k), localDestination);
        CHECK_CUDA(cudaMemcpy2DAsync(
          A(k, k), ldak * sizeof(double), localDestination,
          ldak * sizeof(double), tempkn * sizeof(double), tempkn, kindOut,
          plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldak * sizeof(double) * tempkn;
        CHECK_CUDA(cudaMemcpyAsync(&info, infoDevice, sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   plasma->cuda_stream[rank]));
        volumeGPU2CPU += sizeof(int);
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        if (info != 0) {
          plasma_request_fail(sequence, request, info + A.nb * k);
          ss_abort();
        }
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(k, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination, A(k, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              kindLocal, plasma->cuda_stream[rank]));
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        start = get_current_time();
        auto kindA = deduceMemcpyKind(bufferA, A(k, n));
        CHECK_CUDA(cudaMemcpy2DAsync(
          bufferA, ldak * sizeof(double), A(k, n), ldak * sizeof(double),
          tempkn * sizeof(double), A.nb, kindA, plasma->cuda_stream[rank]));
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        volumeCPU2GPU += ldak * sizeof(double) * A.nb;
        end = get_current_time();
        log_event(rank, device, EVENT_C2G, start, end);

        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDsyrk(plasma->cublas_handle[rank],
                                   CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tempkn,
                                   A.nb, &none, bufferA, ldak, &one,
                                   localDestination, ldak));
          //                    CORE_dsyrk(
          //                         PlasmaLower, PlasmaNoTrans,
          //                         tempkn, A.nb,
          //                         -1.0, A(k, n), ldak,
          //                          1.0, A(k, k), ldak);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dsyrk(PlasmaUpper, PlasmaTrans, tempkn, A.nb, -1.0, A(n, k),
                     ldan, 1.0, A(k, k), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    } else {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination, A(m, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindLocal,
              plasma->cuda_stream[rank]));
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, k, 1);
        if (kDiag != k) {
          start = get_current_time();
            auto kindDiag = deduceMemcpyKind(diagonal, A(k, k));
            CHECK_CUDA(cudaMemcpy2DAsync(
              diagonal, ldak * sizeof(double), A(k, k), ldak * sizeof(double),
              A.nb * sizeof(double), A.nb, kindDiag, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          kDiag = k;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }

        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDtrsm(
              plasma->cublas_handle[rank], CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tempmn,
              A.nb, &zone, diagonal, ldak, localDestination, ldam));
          //                    CORE_dtrsm(
          //                        PlasmaRight, PlasmaLower, PlasmaTrans,
          //                        PlasmaNonUnit, tempmn, A.nb, zone, A(k, k),
          //                        ldak,
          //                              A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit, A.nb,
                     tempmn, zone, A(k, k), ldak, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        auto kindBack = deduceMemcpyKind(A(m, k), localDestination);
        CHECK_CUDA(cudaMemcpy2DAsync(
          A(m, k), ldam * sizeof(double), localDestination,
          ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindBack,
          plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldam * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(m, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination, A(m, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination, ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindLocal,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        start = get_current_time();
        auto kindB = deduceMemcpyKind(bufferB, A(k, n));
        CHECK_CUDA(cudaMemcpy2DAsync(
          bufferB, ldak * sizeof(double), A(k, n), ldak * sizeof(double),
          A.nb * sizeof(double), A.nb, kindB, plasma->cuda_stream[rank]));
        volumeCPU2GPU += ldak * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_C2G, start, end);

        ss_cond_wait(m, n, 1);
        start = get_current_time();
        auto kindA2 = deduceMemcpyKind(bufferA, A(m, n));
        CHECK_CUDA(cudaMemcpy2DAsync(
          bufferA, ldam * sizeof(double), A(m, n), ldam * sizeof(double),
          tempmn * sizeof(double), A.nb, kindA2,
          plasma->cuda_stream[rank]));
        volumeCPU2GPU += ldam * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_C2G, start, end);
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDgemm(plasma->cublas_handle[rank], CUBLAS_OP_N,
                                   CUBLAS_OP_T, tempmn, A.nb, A.nb, &mzone,
                                   bufferA, ldam, bufferB, ldak, &zone,
                                   localDestination, ldam));
          //                    CORE_dgemm(
          //                        PlasmaNoTrans, PlasmaTrans,
          //                        tempmn, A.nb, A.nb,
          //                        mzone, A(m, n), ldam,
          //                               A(k, n), ldak,
          //                         zone, A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dgemm(PlasmaTrans, PlasmaNoTrans, A.nb, tempmn, A.nb, mzone,
                     A(n, k), ldan, A(n, m), ldan, zone, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  CHECK_CUDA(cudaFree(pBuffer));
  CHECK_CUDA(cudaFree(buffer));
  ss_finalize();
#undef volumeCPU2GPU
#undef volumeGPU2CPU
#undef A
#define A(m, n) BLKADDR(A, double, m, n)
}

void plasma_pdpotrf_gpu_reuse_data_table_dist(plasma_context_t *plasma) {
#undef A
#define A(m, n) A_AFFINITY(m, n)
  const int rank = PLASMA_RANK;
  const int device = plasma->rankMapping[rank];
  cudaSetDevice(device);
  double start, end;
  if (rank == 0) {
    plasma->volumeCPU2GPU = 0;
    plasma->volumeGPU2CPU = 0;
  }

  if (rank == plasma->rootRankOfDevice[device]) {
    ((DataTable **)plasma->dataTable)[device] = new DataTable{};
  }
  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  if (uplo == PlasmaUpper) {
    printf("Upper is not supported yet\n");
    return;
  }
  if (sequence->status != PLASMA_SUCCESS) return;
#define volumeCPU2GPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeCPU2GPU)> &>( \
      plasma->volumeCPU2GPU))
#define volumeGPU2CPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeGPU2CPU)> &>( \
      plasma->volumeGPU2CPU))

  k = 0;
  m = PLASMA_RANK;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;

  size_t workspaceInBytes;
  CHECK_CUSOLVER(cusolverDnPotrf_bufferSize(
      plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
      CUBLAS_FILL_MODE_LOWER, A.nb, CUDA_R_64F, NULL, A.nb, CUDA_R_64F,
      &workspaceInBytes));
  //  double *buffer, *bufferA, *bufferB, *bufferC;
  const size_t bufferSizeInByte = (unsigned long)(A.nb) * A.nb * sizeof(double);
  std::shared_ptr<double> localDestination{
      (double *)cudaAllocator(bufferSizeInByte), cudaDeleter};
  //  cudaMalloc((void **)&buffer, 3 * bufferSizeInByte);
  //  bufferA = buffer;
  //  bufferB = bufferA + (unsigned long)(A.nb) * A.nb;
  //  bufferC = bufferB + (unsigned long)(A.nb) * A.nb;
  void *pBuffer;
  CHECK_CUDA(
      cudaMalloc(&pBuffer, workspaceInBytes +
                               (sizeof(int) - workspaceInBytes % sizeof(int)) +
                               sizeof(int)));
  infoDevice = (int *)((long)pBuffer + workspaceInBytes +
                       (sizeof(int) - workspaceInBytes % sizeof(int)));
  double one = 1., none = -1.;
  ss_init(A.nt, A.nt, 0);
  // synced in ss_init
  auto dataTable = ((DataTable **)plasma->dataTable)[device];
  if (rank == plasma->rootRankOfDevice[device]) {
    // Do not have to sync because the first potrf is done by rank-0
    dataTable->diagonal.reset((double *)cudaAllocator(bufferSizeInByte),
                              cudaDeleter);
    const auto freeBlocks = getMapBound(A.nb);
    // Do not have to sync because the first potrf is done by rank-0
    dataTable->mapBound =
        max(freeBlocks / plasma->uniqueDevice,
            min(plasma->uniqueDevice * 3, freeBlocks));
    if (dataTable->mapBound < plasma->uniqueDevice * 3) {
      printf("GPU out of memory, mostly because NB is too large\n");
      ss_abort();
    }
  }

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination.get(), A(k, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              kindLocal, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUSOLVER(cusolverDnPotrf(
              plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
              CUBLAS_FILL_MODE_LOWER, tempkn, CUDA_R_64F,
              localDestination.get(), ldak, CUDA_R_64F, pBuffer,
              workspaceInBytes, infoDevice));
          //                    CORE_dpotrf(
          //                        PlasmaLower,
          //                        tempkn,
          //                        A(k, k), ldak,
          //                        &info);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dpotrf(PlasmaUpper, tempkn, A(k, k), ldak, &info);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        auto kindOut = deduceMemcpyKind(A(k, k), localDestination.get());
        CHECK_CUDA(cudaMemcpy2DAsync(
          A(k, k), ldak * sizeof(double), localDestination.get(),
          ldak * sizeof(double), tempkn * sizeof(double), tempkn, kindOut,
          plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldak * sizeof(double) * tempkn;
        CHECK_CUDA(cudaMemcpyAsync(&info, infoDevice, sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   plasma->cuda_stream[rank]));
        volumeGPU2CPU += sizeof(int);
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        std::swap(localDestination, dataTable->diagonal);
        dataTable->diagIdx = k;
        lock.unlock();
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        while (localDestination.use_count() > 1) {
          plasma_yield();
        }
        if (info != 0) {
          plasma_request_fail(sequence, request, info + A.nb * k);
          ss_abort();
        }
        ss_cond_set(k, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination.get(), A(k, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              kindLocal, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        auto [bufferA, existA, landedA] = lookupTable(
            dataTable->map, dataTable->mapBound, {k, n}, bufferSizeInByte);
        lock.unlock();
        if (!existA) {
          start = get_current_time();
            auto kindA = deduceMemcpyKind(bufferA.get(), A(k, n));
            CHECK_CUDA(cudaMemcpy2DAsync(
              bufferA.get(), ldak * sizeof(double), A(k, n),
              ldak * sizeof(double), tempkn * sizeof(double), A.nb, kindA,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedA = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }

        while (!*landedA) {
          plasma_yield();
        }

        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDsyrk(plasma->cublas_handle[rank],
                                   CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tempkn,
                                   A.nb, &none, bufferA.get(), ldak, &one,
                                   localDestination.get(), ldak));
          //                    CORE_dsyrk(
          //                         PlasmaLower, PlasmaNoTrans,
          //                         tempkn, A.nb,
          //                         -1.0, A(k, n), ldak,
          //                          1.0, A(k, k), ldak);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dsyrk(PlasmaUpper, PlasmaTrans, tempkn, A.nb, -1.0, A(n, k),
                     ldan, 1.0, A(k, k), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    } else {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination.get(), A(m, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindLocal,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, k, 1);
        bool *landed = nullptr;
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        std::shared_ptr<double> diagonal = dataTable->diagonal;
        lock.unlock();
        if (dataTable->diagIdx != k /* will data racing? */) {
          lock.lock();
          auto [bufferA, existA, landedA] = lookupTable(
              dataTable->map, dataTable->mapBound, {k, k}, bufferSizeInByte);
          lock.unlock();
          landed = landedA;
          if (!existA) {
            start = get_current_time();
            auto kindDiag = deduceMemcpyKind(bufferA.get(), A(k, k));
            CHECK_CUDA(cudaMemcpy2DAsync(
              bufferA.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), A.nb * sizeof(double), A.nb, kindDiag,
              plasma->cuda_stream[rank]));
            volumeCPU2GPU += ldak * sizeof(double) * A.nb;
            CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
            *landedA = true;
            end = get_current_time();
            log_event(rank, device, EVENT_C2G, start, end);
          }
          diagonal = std::move(bufferA);
        }
        if (landed != nullptr) {
          while (!*landed) {
            plasma_yield();
          }
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDtrsm(
              plasma->cublas_handle[rank], CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tempmn,
              A.nb, &zone, diagonal.get(), ldak, localDestination.get(), ldam));
          //                    CORE_dtrsm(
          //                        PlasmaRight, PlasmaLower, PlasmaTrans,
          //                        PlasmaNonUnit, tempmn, A.nb, zone, A(k, k),
          //                        ldak,
          //                              A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit, A.nb,
                     tempmn, zone, A(k, k), ldak, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        auto kindBack = deduceMemcpyKind(A(m, k), localDestination.get());
        CHECK_CUDA(cudaMemcpy2DAsync(
          A(m, k), ldam * sizeof(double), localDestination.get(),
          ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindBack,
          plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldam * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(m, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          auto kindLocal = deduceMemcpyKind(localDestination.get(), A(m, k));
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindLocal,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        auto [bufferB, existB, landedB] = lookupTable(
            dataTable->map, dataTable->mapBound, {k, n}, bufferSizeInByte);
        lock.unlock();
        if (!existB) {
          start = get_current_time();
            auto kindB = deduceMemcpyKind(bufferB.get(), A(k, n));
            CHECK_CUDA(cudaMemcpy2DAsync(
              bufferB.get(), ldak * sizeof(double), A(k, n),
              ldak * sizeof(double), A.nb * sizeof(double), A.nb, kindB,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedB = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(m, n, 1);
        lock.lock();
        auto [bufferC, existC, landedC] = lookupTable(
            dataTable->map, dataTable->mapBound, {m, n}, bufferSizeInByte);
        lock.unlock();
        if (!existC) {
          start = get_current_time();
            auto kindC = deduceMemcpyKind(bufferC.get(), A(m, n));
            CHECK_CUDA(cudaMemcpy2DAsync(
              bufferC.get(), ldam * sizeof(double), A(m, n),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb, kindC,
              plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedC = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        while (!*landedB) {
          plasma_yield();
        }
        while (!*landedC) {
          plasma_yield();
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDgemm(plasma->cublas_handle[rank], CUBLAS_OP_N,
                                   CUBLAS_OP_T, tempmn, A.nb, A.nb, &mzone,
                                   bufferC.get(), ldam, bufferB.get(), ldak,
                                   &zone, localDestination.get(), ldam));
          //                    CORE_dgemm(
          //                        PlasmaNoTrans, PlasmaTrans,
          //                        tempmn, A.nb, A.nb,
          //                        mzone, A(m, n), ldam,
          //                               A(k, n), ldak,
          //                         zone, A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dgemm(PlasmaTrans, PlasmaNoTrans, A.nb, tempmn, A.nb, mzone,
                     A(n, k), ldan, A(n, m), ldan, zone, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  CHECK_CUDA(cudaFree(pBuffer));
  //  CHECK_CUDA(cudaFree(buffer));
  ss_finalize();
  if (rank == plasma->rootRankOfDevice[device]) {
    delete dataTable;
  }
#undef volumeCPU2GPU
#undef volumeGPU2CPU
#undef A
#define A(m, n) BLKADDR(A, double, m, n)
}

void plasma_pdpotrf_gpu_reuse_data_table_all_managed_dist(
    plasma_context_t *plasma) {
#undef A
#define A(m, n) A_AFFINITY(m, n)
  const int rank = PLASMA_RANK;
  const int device = plasma->rankMapping[rank];
  double start, end;
  cudaSetDevice(device);
  if (rank == 0) {
    plasma->volumeCPU2GPU = 0;
    plasma->volumeGPU2CPU = 0;
  }

  if (rank == plasma->rootRankOfDevice[device]) {
    ((DataTable **)plasma->dataTable)[device] = new DataTable{};
  }

  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  if (uplo == PlasmaUpper) {
    printf("Upper is not supported yet\n");
    return;
  }
  if (sequence->status != PLASMA_SUCCESS) return;
#define volumeCPU2GPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeCPU2GPU)> &>( \
      plasma->volumeCPU2GPU))
#define volumeGPU2CPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeGPU2CPU)> &>( \
      plasma->volumeGPU2CPU))

  k = 0;
  m = PLASMA_RANK;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;

  size_t workspaceInBytes;
  CHECK_CUSOLVER(cusolverDnPotrf_bufferSize(
      plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
      CUBLAS_FILL_MODE_LOWER, A.nb, CUDA_R_64F, NULL, A.nb, CUDA_R_64F,
      &workspaceInBytes));
  //  double *buffer, *bufferA, *bufferB, *bufferC;
  const size_t bufferSizeInByte = (unsigned long)(A.nb) * A.nb * sizeof(double);
  std::shared_ptr<double> localDestination;
  //  cudaMalloc((void **)&buffer, 3 * bufferSizeInByte);
  //  bufferA = buffer;
  //  bufferB = bufferA + (unsigned long)(A.nb) * A.nb;
  //  bufferC = bufferB + (unsigned long)(A.nb) * A.nb;
  void *pBuffer;
  CHECK_CUDA(
      cudaMalloc(&pBuffer, workspaceInBytes +
                               (sizeof(int) - workspaceInBytes % sizeof(int)) +
                               sizeof(int)));
  infoDevice = (int *)((long)pBuffer + workspaceInBytes +
                       (sizeof(int) - workspaceInBytes % sizeof(int)));
  double one = 1., none = -1.;
  ss_init(A.nt, A.nt, 0);
  // synced in ss_init
  auto dataTable = ((DataTable **)plasma->dataTable)[device];
  if (rank == plasma->rootRankOfDevice[device]) {
    const auto freeBlocks = getMapBound(A.nb);
    // Do not have to sync because the first potrf is done by rank-0
    dataTable->mapBound =
        max(freeBlocks / plasma->uniqueDevice,
            min(plasma->uniqueDevice * 3, freeBlocks));
    if (dataTable->mapBound < plasma->uniqueDevice * 3) {
      printf("GPU out of memory, mostly because NB is too large\n");
      ss_abort();
    }
  }
  plasma_barrier(plasma);

  // Initialize localDestination
  {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }
    if (k == 0) {
      std::unique_lock<std::mutex> lock{dataTable->mutex};
      auto [bufferDestination, _1, _2] = lookupTable(
          dataTable->map, dataTable->mapBound, {m, k}, bufferSizeInByte);
      lock.unlock();
      localDestination = std::move(bufferDestination);
    } else {
      std::unique_lock<std::mutex> lock{dataTable->mutex};
      auto [bufferDestination, _1, _2] =
          lookupTable(dataTable->map, dataTable->mapBound, {next_m, next_k},
                      bufferSizeInByte);
      lock.unlock();
      localDestination = std::move(bufferDestination);
    }
  }

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUSOLVER(cusolverDnPotrf(
              plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
              CUBLAS_FILL_MODE_LOWER, tempkn, CUDA_R_64F,
              localDestination.get(), ldak, CUDA_R_64F, pBuffer,
              workspaceInBytes, infoDevice));
          //                    CORE_dpotrf(
          //                        PlasmaLower,
          //                        tempkn,
          //                        A(k, k), ldak,
          //                        &info);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dpotrf(PlasmaUpper, tempkn, A(k, k), ldak, &info);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        CHECK_CUDA(cudaMemcpy2DAsync(
            A(k, k), ldak * sizeof(double), localDestination.get(),
            ldak * sizeof(double), tempkn * sizeof(double), tempkn,
            cudaMemcpyDeviceToHost, plasma->cuda_stream[rank]));
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        volumeGPU2CPU += ldak * sizeof(double) * tempkn;
        CHECK_CUDA(cudaMemcpyAsync(&info, infoDevice, sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   plasma->cuda_stream[rank]));
        volumeGPU2CPU += sizeof(int);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        dataTable->diagIdx = k;
        dataTable->diagonal = std::move(localDestination);
        setLandedTable(dataTable->map, {k, k}, true);
        auto [bufferNext, existNext, landedNext] =
            lookupTable(dataTable->map, dataTable->mapBound, {next_m, next_k},
                        bufferSizeInByte);
        lock.unlock();
        if (existNext) {
          printf(
              "Unexpected situation happened at %s:%d, please report the bug\n",
              __FILE__, __LINE__);
          ss_abort();
        }
        localDestination = std::move(bufferNext);
        if (info != 0) {
          plasma_request_fail(sequence, request, info + A.nb * k);
          ss_abort();
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        ss_cond_set(k, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        auto [bufferA, existA, landedA] = lookupTable(
            dataTable->map, dataTable->mapBound, {k, n}, bufferSizeInByte);
        lock.unlock();
        if (!existA) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              bufferA.get(), ldak * sizeof(double), A(k, n),
              ldak * sizeof(double), tempkn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          *landedA = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }

        while (!*landedA) {
          plasma_yield();
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDsyrk(plasma->cublas_handle[rank],
                                   CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tempkn,
                                   A.nb, &none, bufferA.get(), ldak, &one,
                                   localDestination.get(), ldak));
          //                    CORE_dsyrk(
          //                         PlasmaLower, PlasmaNoTrans,
          //                         tempkn, A.nb,
          //                         -1.0, A(k, n), ldak,
          //                          1.0, A(k, k), ldak);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dsyrk(PlasmaUpper, PlasmaTrans, tempkn, A.nb, -1.0, A(n, k),
                     ldan, 1.0, A(k, k), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    } else {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, k, 1);
        bool *landed = nullptr;
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        std::shared_ptr<double> diagonal = dataTable->diagonal;
        lock.unlock();
        if (dataTable->diagIdx != k /* will data racing? */) {
          lock.lock();
          auto [bufferA, existA, landedA] = lookupTable(
              dataTable->map, dataTable->mapBound, {k, k}, bufferSizeInByte);
          lock.unlock();
          landed = landedA;
          if (!existA) {
            start = get_current_time();
            CHECK_CUDA(cudaMemcpy2DAsync(
                bufferA.get(), ldak * sizeof(double), A(k, k),
                ldak * sizeof(double), A.nb * sizeof(double), A.nb,
                cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
            volumeCPU2GPU += ldak * sizeof(double) * A.nb;
            CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
            *landedA = true;
            end = get_current_time();
            log_event(rank, device, EVENT_C2G, start, end);
          }
          diagonal = std::move(bufferA);
        }
        if (landed != nullptr) {
          while (!*landed) {
            plasma_yield();
          }
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDtrsm(
              plasma->cublas_handle[rank], CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tempmn,
              A.nb, &zone, diagonal.get(), ldak, localDestination.get(), ldam));
          //                    CORE_dtrsm(
          //                        PlasmaRight, PlasmaLower, PlasmaTrans,
          //                        PlasmaNonUnit, tempmn, A.nb, zone, A(k, k),
          //                        ldak,
          //                              A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit, A.nb,
                     tempmn, zone, A(k, k), ldak, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        CHECK_CUDA(cudaMemcpy2DAsync(
            A(m, k), ldam * sizeof(double), localDestination.get(),
            ldam * sizeof(double), tempmn * sizeof(double), A.nb,
            cudaMemcpyDeviceToHost, plasma->cuda_stream[rank]));
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldam * sizeof(double) * A.nb;
        lock.lock();
        setLandedTable(dataTable->map, {m, k}, true);
        auto [bufferNext, existNext, landedNext] =
            lookupTable(dataTable->map, dataTable->mapBound, {next_m, next_k},
                        bufferSizeInByte);
        lock.unlock();
        if (existNext) {
          printf(
              "Unexpected situation happened at %s:%d, please report the bug\n",
              __FILE__, __LINE__);
          ss_abort();
        }
        localDestination = std::move(bufferNext);

        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(m, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        auto [bufferB, existB, landedB] = lookupTable(
            dataTable->map, dataTable->mapBound, {k, n}, bufferSizeInByte);
        lock.unlock();
        if (!existB) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              bufferB.get(), ldak * sizeof(double), A(k, n),
              ldak * sizeof(double), A.nb * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedB = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(m, n, 1);
        lock.lock();
        auto [bufferC, existC, landedC] = lookupTable(
            dataTable->map, dataTable->mapBound, {m, n}, bufferSizeInByte);
        lock.unlock();
        if (!existC) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              bufferC.get(), ldam * sizeof(double), A(m, n),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedC = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        while (!*landedB) {
          plasma_yield();
        }
        while (!*landedC) {
          plasma_yield();
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDgemm(plasma->cublas_handle[rank], CUBLAS_OP_N,
                                   CUBLAS_OP_T, tempmn, A.nb, A.nb, &mzone,
                                   bufferC.get(), ldam, bufferB.get(), ldak,
                                   &zone, localDestination.get(), ldam));
          //                    CORE_dgemm(
          //                        PlasmaNoTrans, PlasmaTrans,
          //                        tempmn, A.nb, A.nb,
          //                        mzone, A(m, n), ldam,
          //                               A(k, n), ldak,
          //                         zone, A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dgemm(PlasmaTrans, PlasmaNoTrans, A.nb, tempmn, A.nb, mzone,
                     A(n, k), ldan, A(n, m), ldan, zone, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  CHECK_CUDA(cudaFree(pBuffer));
  //  CHECK_CUDA(cudaFree(buffer));
  ss_finalize();
  if (rank == plasma->rootRankOfDevice[device]) {
    delete dataTable;
  }
#undef volumeCPU2GPU
#undef volumeGPU2CPU
#undef A
#define A(m, n) BLKADDR(A, double, m, n)
}

void plasma_pdpotrf_gpu_reuse_data_table_no_free_dist(
    plasma_context_t *plasma) {
#undef A
#define A(m, n) A_AFFINITY(m, n)
  const int rank = PLASMA_RANK;
  const int device = plasma->rankMapping[rank];
  double start, end;
  cudaSetDevice(device);
  if (rank == 0) {
    plasma->volumeCPU2GPU = 0;
    plasma->volumeGPU2CPU = 0;
  }

  if (rank == plasma->rootRankOfDevice[device]) {
    ((DataTable **)plasma->dataTable)[device] = new DataTable{};
  }

  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  if (uplo == PlasmaUpper) {
    printf("Upper is not supported yet\n");
    return;
  }
  if (sequence->status != PLASMA_SUCCESS) return;
#define volumeCPU2GPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeCPU2GPU)> &>( \
      plasma->volumeCPU2GPU))
#define volumeGPU2CPU                                                \
  (reinterpret_cast<std::atomic<decltype(plasma->volumeGPU2CPU)> &>( \
      plasma->volumeGPU2CPU))

  k = 0;
  m = PLASMA_RANK;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;

  size_t workspaceInBytes;
  CHECK_CUSOLVER(cusolverDnPotrf_bufferSize(
      plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
      CUBLAS_FILL_MODE_LOWER, A.nb, CUDA_R_64F, NULL, A.nb, CUDA_R_64F,
      &workspaceInBytes));
  //  double *buffer, *bufferA, *bufferB, *bufferC;
  const size_t bufferSizeInByte = (unsigned long)(A.nb) * A.nb * sizeof(double);
  std::shared_ptr<double> localDestination;
  //  cudaMalloc((void **)&buffer, 3 * bufferSizeInByte);
  //  bufferA = buffer;
  //  bufferB = bufferA + (unsigned long)(A.nb) * A.nb;
  //  bufferC = bufferB + (unsigned long)(A.nb) * A.nb;
  void *pBuffer;
  CHECK_CUDA(
      cudaMalloc(&pBuffer, workspaceInBytes +
                               (sizeof(int) - workspaceInBytes % sizeof(int)) +
                               sizeof(int)));
  infoDevice = (int *)((long)pBuffer + workspaceInBytes +
                       (sizeof(int) - workspaceInBytes % sizeof(int)));
  double one = 1., none = -1.;
  ss_init(A.nt, A.nt, 0);
  // synced in ss_init
  auto dataTable = ((DataTable **)plasma->dataTable)[device];
  if (rank == plasma->rootRankOfDevice[device]) {
    const auto freeBlocks = getMapBound(A.nb);
    // Do not have to sync because the first potrf is done by rank-0
    dataTable->mapBound = max(freeBlocks / plasma->uniqueDevice,
                              min(plasma->world_size, freeBlocks));
    if (dataTable->mapBound < plasma->uniqueDevice * 3) {
      printf("GPU out of memory, mostly because NB is too large\n");
      ss_abort();
    }
  }
  plasma_barrier(plasma);

  // Initialize localDestination
  {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }
    if (k == 0) {
      std::unique_lock<std::mutex> lock{dataTable->mutex};
      auto [bufferDestination, _1, _2] = lookupTable(
          dataTable->map, dataTable->mapBound, {m, k}, bufferSizeInByte);
      lock.unlock();
      localDestination = std::move(bufferDestination);
    } else {
      std::unique_lock<std::mutex> lock{dataTable->mutex};
      auto [bufferDestination, _1, _2] =
          lookupTable(dataTable->map, dataTable->mapBound, {next_m, next_k},
                      bufferSizeInByte);
      lock.unlock();
      localDestination = std::move(bufferDestination);
    }
  }

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUSOLVER(cusolverDnPotrf(
              plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
              CUBLAS_FILL_MODE_LOWER, tempkn, CUDA_R_64F,
              localDestination.get(), ldak, CUDA_R_64F, pBuffer,
              workspaceInBytes, infoDevice));
          //                    CORE_dpotrf(
          //                        PlasmaLower,
          //                        tempkn,
          //                        A(k, k), ldak,
          //                        &info);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dpotrf(PlasmaUpper, tempkn, A(k, k), ldak, &info);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        CHECK_CUDA(cudaMemcpy2DAsync(
            A(k, k), ldak * sizeof(double), localDestination.get(),
            ldak * sizeof(double), tempkn * sizeof(double), tempkn,
            cudaMemcpyDeviceToHost, plasma->cuda_stream[rank]));
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        volumeGPU2CPU += ldak * sizeof(double) * tempkn;
        CHECK_CUDA(cudaMemcpyAsync(&info, infoDevice, sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   plasma->cuda_stream[rank]));
        volumeGPU2CPU += sizeof(int);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        dataTable->diagIdx = k;
        dataTable->diagonal = std::move(localDestination);
        setLandedTable(dataTable->map, {k, k}, true);
        auto [bufferNext, existNext, landedNext] =
            lookupTable(dataTable->map, dataTable->mapBound, {next_m, next_k},
                        bufferSizeInByte, dataTable->canAllocateNew);
        lock.unlock();
        if (existNext) {
          printf(
              "Unexpected situation happened at %s:%d, please report the bug\n",
              __FILE__, __LINE__);
          ss_abort();
        }
        localDestination = std::move(bufferNext);
        if (info != 0) {
          plasma_request_fail(sequence, request, info + A.nb * k);
          ss_abort();
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        lock.lock();
        int remainingSize = 1 + (k + 2 + A.nt) * (A.nt - k - 1) / 2 -
                            (long)dataTable->map.size();
        if (remainingSize < 0 && dataTable->canAllocateNew) {
          dataTable->canAllocateNew = false;
        }
        lock.unlock();
        ss_cond_set(k, k, 1);
        // TODO(Jie): maybe use size_t
        if (remainingSize < 0) {
          if (dataTable->canAllocateNew) {
            for (int i = 0; i < k; ++i) {
              for (int j = 0; j <= i; ++j) {
                lock.lock();
                auto find =
                    dataTable->map.find(DataTable::Coord{.pair = {i, j}}.dummy);
                if (find != dataTable->map.end()) {
                }
                lock.unlock();
              }
            }
          } else {
            for (int j = 0; j <= k; ++j) {
              lock.lock();
              auto find = dataTable->map.find(
                  DataTable::Coord{.pair = {k - 1, j}}.dummy);
              if (find != dataTable->map.end()) {
                dataTable->map.erase(find);
              }
              lock.unlock();
            }
          }
        }
      } else {
        if (n == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldak * sizeof(double), A(k, k),
              ldak * sizeof(double), tempkn * sizeof(double), tempkn,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        auto [bufferA, existA, landedA] =
            lookupTable(dataTable->map, dataTable->mapBound, {k, n},
                        bufferSizeInByte, dataTable->canAllocateNew);
        lock.unlock();
        if (!existA) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              bufferA.get(), ldak * sizeof(double), A(k, n),
              ldak * sizeof(double), tempkn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          *landedA = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }

        while (!*landedA) {
          plasma_yield();
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDsyrk(plasma->cublas_handle[rank],
                                   CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tempkn,
                                   A.nb, &none, bufferA.get(), ldak, &one,
                                   localDestination.get(), ldak));
          //                    CORE_dsyrk(
          //                         PlasmaLower, PlasmaNoTrans,
          //                         tempkn, A.nb,
          //                         -1.0, A(k, n), ldak,
          //                          1.0, A(k, k), ldak);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dsyrk(PlasmaUpper, PlasmaTrans, tempkn, A.nb, -1.0, A(n, k),
                     ldan, 1.0, A(k, k), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    } else {
      if (n == k) {
        if (k == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, k, 1);
        bool *landed = nullptr;
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        std::shared_ptr<double> diagonal = dataTable->diagonal;
        lock.unlock();
        if (dataTable->diagIdx != k /* will data racing? */) {
          lock.lock();
          auto [bufferA, existA, landedA] =
              lookupTable(dataTable->map, dataTable->mapBound, {k, k},
                          bufferSizeInByte, dataTable->canAllocateNew);
          lock.unlock();
          landed = landedA;
          if (!existA) {
            start = get_current_time();
            CHECK_CUDA(cudaMemcpy2DAsync(
                bufferA.get(), ldak * sizeof(double), A(k, k),
                ldak * sizeof(double), A.nb * sizeof(double), A.nb,
                cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
            volumeCPU2GPU += ldak * sizeof(double) * A.nb;
            CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
            *landedA = true;
            end = get_current_time();
            log_event(rank, device, EVENT_C2G, start, end);
          }
          diagonal = std::move(bufferA);
        }
        if (landed != nullptr) {
          while (!*landed) {
            plasma_yield();
          }
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDtrsm(
              plasma->cublas_handle[rank], CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tempmn,
              A.nb, &zone, diagonal.get(), ldak, localDestination.get(), ldam));
          //                    CORE_dtrsm(
          //                        PlasmaRight, PlasmaLower, PlasmaTrans,
          //                        PlasmaNonUnit, tempmn, A.nb, zone, A(k, k),
          //                        ldak,
          //                              A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit, A.nb,
                     tempmn, zone, A(k, k), ldak, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
        start = get_current_time();
        CHECK_CUDA(cudaMemcpy2DAsync(
            A(m, k), ldam * sizeof(double), localDestination.get(),
            ldam * sizeof(double), tempmn * sizeof(double), A.nb,
            cudaMemcpyDeviceToHost, plasma->cuda_stream[rank]));
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        volumeGPU2CPU += ldam * sizeof(double) * A.nb;
        lock.lock();
        setLandedTable(dataTable->map, {m, k}, true);
        auto [bufferNext, existNext, landedNext] =
            lookupTable(dataTable->map, dataTable->mapBound, {next_m, next_k},
                        bufferSizeInByte, dataTable->canAllocateNew);
        lock.unlock();
        if (existNext) {
          printf(
              "Unexpected situation happened at %s:%d, please report the bug\n",
              __FILE__, __LINE__);
          ss_abort();
        }
        localDestination = std::move(bufferNext);

        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_G2C, start, end);
        ss_cond_set(m, k, 1);
      } else {
        if (n == 0) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              localDestination.get(), ldam * sizeof(double), A(m, k),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(k, n, 1);
        std::unique_lock<std::mutex> lock{dataTable->mutex};
        auto [bufferB, existB, landedB] =
            lookupTable(dataTable->map, dataTable->mapBound, {k, n},
                        bufferSizeInByte, dataTable->canAllocateNew);
        lock.unlock();
        if (!existB) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              bufferB.get(), ldak * sizeof(double), A(k, n),
              ldak * sizeof(double), A.nb * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedB = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        ss_cond_wait(m, n, 1);
        lock.lock();
        auto [bufferC, existC, landedC] =
            lookupTable(dataTable->map, dataTable->mapBound, {m, n},
                        bufferSizeInByte, dataTable->canAllocateNew);
        lock.unlock();
        if (!existC) {
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              bufferC.get(), ldam * sizeof(double), A(m, n),
              ldam * sizeof(double), tempmn * sizeof(double), A.nb,
              cudaMemcpyHostToDevice, plasma->cuda_stream[rank]));
          volumeCPU2GPU += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
          *landedC = true;
          end = get_current_time();
          log_event(rank, device, EVENT_C2G, start, end);
        }
        while (!*landedB) {
          plasma_yield();
        }
        while (!*landedC) {
          plasma_yield();
        }
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDgemm(plasma->cublas_handle[rank], CUBLAS_OP_N,
                                   CUBLAS_OP_T, tempmn, A.nb, A.nb, &mzone,
                                   bufferC.get(), ldam, bufferB.get(), ldak,
                                   &zone, localDestination.get(), ldam));
          //                    CORE_dgemm(
          //                        PlasmaNoTrans, PlasmaTrans,
          //                        tempmn, A.nb, A.nb,
          //                        mzone, A(m, n), ldam,
          //                               A(k, n), ldak,
          //                         zone, A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dgemm(PlasmaTrans, PlasmaNoTrans, A.nb, tempmn, A.nb, mzone,
                     A(n, k), ldan, A(n, m), ldan, zone, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, device, EVENT_COMPUTE, start, end);
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  CHECK_CUDA(cudaFree(pBuffer));
  //  CHECK_CUDA(cudaFree(buffer));
  ss_finalize();
  if (rank == plasma->rootRankOfDevice[device]) {
    delete dataTable;
  }
#undef volumeCPU2GPU
#undef volumeGPU2CPU
#undef A
#define A(m, n) BLKADDR(A, double, m, n)
}

#ifdef PLASMA_WITH_HPCASIA24

namespace {
struct MemoryStatus {
  double *ptr = nullptr;
  std::atomic<int> count = 0;
};
}  // namespace

#define ms_get_ptr(m, n) (memoryStatus[(m) + plasma->ss_ld * (n)].ptr)
#define ms_get_count(m, n) (memoryStatus[(m) + plasma->ss_ld * (n)].count)

#define ms_cond_wait(m, n, val)                                            \
  {                                                                        \
    while (!plasma->ss_abort && ms_get_ptr(m, n) == (val)) plasma_yield(); \
    if (plasma->ss_abort) break;                                           \
  }

void plasma_pdpotrf_gpu_reuse_data_static_table_computation_dist(
    const int rank, const int rankDev, int *current_step, void *pBuffer,
    size_t workspaceInBytes, int *infoDevice, MemoryStatus *memoryStatus,
    plasma_context_t *plasma) {
  double start, end;
  cudaSetDevice(rankDev);

  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  // synced in ss_init

  k = 0;
  m = rank;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;

  double one = 1., none = -1.;

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        ms_cond_wait(k, k, nullptr);
        start = get_current_time();
        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUSOLVER(cusolverDnPotrf(
              plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
              CUBLAS_FILL_MODE_LOWER, tempkn, CUDA_R_64F, ms_get_ptr(k, k),
              ldak, CUDA_R_64F, pBuffer, workspaceInBytes, infoDevice));
          //                    CORE_dpotrf(
          //                        PlasmaLower,
          //                        tempkn,
          //                        A(k, k), ldak,
          //                        &info);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dpotrf(PlasmaUpper, tempkn, A(k, k), ldak, &info);
        }
        CHECK_CUDA(cudaMemcpyAsync(&info, infoDevice, sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   plasma->cuda_stream[rank]));
        make_atomic(plasma->volumeGPU2CPU) += sizeof(int);
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, rankDev, EVENT_COMPUTE, start, end);
        if (info != 0) {
          plasma_request_fail(sequence, request, info + A.nb * k);
          ss_abort();
        }
        ++*current_step;
      } else {
        ms_cond_wait(k, k, nullptr);
        ms_cond_wait(k, n, nullptr);
        start = get_current_time();

        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDsyrk(plasma->cublas_handle[rank],
                                   CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tempkn,
                                   A.nb, &none, ms_get_ptr(k, n), ldak, &one,
                                   ms_get_ptr(k, k), ldak));
          //                    CORE_dsyrk(
          //                         PlasmaLower, PlasmaNoTrans,
          //                         tempkn, A.nb,
          //                         -1.0, A(k, n), ldak,
          //                          1.0, A(k, k), ldak);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dsyrk(PlasmaUpper, PlasmaTrans, tempkn, A.nb, -1.0, A(n, k),
                     ldan, 1.0, A(k, k), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, rankDev, EVENT_COMPUTE, start, end);
        ++*current_step;
      }
    } else {
      if (n == k) {
        ms_cond_wait(m, k, nullptr);
        ms_cond_wait(k, k, nullptr);
        start = get_current_time();

        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDtrsm(
              plasma->cublas_handle[rank], CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tempmn,
              A.nb, &zone, ms_get_ptr(k, k), ldak, ms_get_ptr(m, k), ldam));
          //                    CORE_dtrsm(
          //                        PlasmaRight, PlasmaLower, PlasmaTrans,
          //                        PlasmaNonUnit, tempmn, A.nb, zone, A(k, k),
          //                        ldak,
          //                              A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit, A.nb,
                     tempmn, zone, A(k, k), ldak, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, rankDev, EVENT_COMPUTE, start, end);
        ++*current_step;
      } else {
        ms_cond_wait(m, k, nullptr);
        ms_cond_wait(k, n, nullptr);
        ms_cond_wait(m, n, nullptr);
        start = get_current_time();

        /*
         *  PlasmaLower
         */
        if (uplo == PlasmaLower) {
          CHECK_CUBLAS(cublasDgemm(plasma->cublas_handle[rank], CUBLAS_OP_N,
                                   CUBLAS_OP_T, tempmn, A.nb, A.nb, &mzone,
                                   ms_get_ptr(m, n), ldam, ms_get_ptr(k, n),
                                   ldak, &zone, ms_get_ptr(m, k), ldam));
          //                    CORE_dgemm(
          //                        PlasmaNoTrans, PlasmaTrans,
          //                        tempmn, A.nb, A.nb,
          //                        mzone, A(m, n), ldam,
          //                               A(k, n), ldak,
          //                         zone, A(m, k), ldam);
        }
        /*
         *  PlasmaUpper
         */
        else {
          CORE_dgemm(PlasmaTrans, PlasmaNoTrans, A.nb, tempmn, A.nb, mzone,
                     A(n, k), ldan, A(n, m), ldan, zone, A(k, m), ldak);
        }
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream[rank]));
        end = get_current_time();
        log_event(rank, rankDev, EVENT_COMPUTE, start, end);
        ++*current_step;
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  // important for releasing g2c!
  ++*current_step;
}

void plasma_pdpotrf_gpu_reuse_data_static_table_g2c_dist(
    const int rank, const int rankDev, MemoryStatus *memoryStatus,
    const int *compute_step, plasma_context_t *plasma) {
  double start, end;
  cudaSetDevice(rankDev);

  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  int current_step = 0;

  plasma_unpack_args_4(uplo, A, sequence, request);
  // synced in ss_init

  k = 0;
  m = rank;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;
  double one = 1., none = -1.;

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    // make sure behind compute step
    while (current_step >= *compute_step) {
      plasma_yield();
    }
    ++current_step;

    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        start = get_current_time();
        CHECK_CUDA(cudaMemcpy2DAsync(
            A(k, k), ldak * sizeof(double), ms_get_ptr(k, k),
            ldak * sizeof(double), tempkn * sizeof(double), tempkn,
            cudaMemcpyDeviceToHost, plasma->cuda_stream_g2c[rank]));
        make_atomic(plasma->volumeGPU2CPU) += ldak * sizeof(double) * tempkn;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_g2c[rank]));
        ms_get_count(k, k)--;
        ss_cond_set(k, k, 1);
        end = get_current_time();
        log_event(rank, rankDev, EVENT_G2C, start, end);
      } else {
        ms_get_count(k, k)--;
        ms_get_count(k, n)--;
      }
    } else {
      if (n == k) {
        start = get_current_time();
        CHECK_CUDA(cudaMemcpy2DAsync(
            A(m, k), ldam * sizeof(double), ms_get_ptr(m, k),
            ldam * sizeof(double), tempmn * sizeof(double), A.nb,
            cudaMemcpyDeviceToHost, plasma->cuda_stream_g2c[rank]));
        make_atomic(plasma->volumeGPU2CPU) += ldam * sizeof(double) * A.nb;
        CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_g2c[rank]));
        ms_get_count(k, k)--;
        ss_cond_set(m, k, 1);
        end = get_current_time();
        log_event(rank, rankDev, EVENT_G2C, start, end);
        ms_get_count(m, k)--;
      } else {
        ms_get_count(m, k)--;
        ms_get_count(k, n)--;
        ms_get_count(m, n)--;
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
}

void plasma_pdpotrf_gpu_reuse_data_static_table_c2g_dist(
    const int rank, const int rankDev, MemoryStatus *memoryStatus,
    plasma_context_t *plasma) {
  double start, end;
  cudaSetDevice(rankDev);

  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  int k, m, n;
  int next_k;
  int next_m;
  int next_n;
  int ldak, ldam, ldan;
  int *infoDevice;
  int info;
  int tempkn, tempmn;

  double zone = (double)1.0;
  double mzone = (double)-1.0;

  plasma_unpack_args_4(uplo, A, sequence, request);

  const int maxBound = getMapBound(A.nb);
  int allocated = 0;

  // synced in ss_init
  const size_t bufferSizeInByte = (unsigned long)(A.nb) * A.nb * sizeof(double);

  k = 0;
  m = rank;
  while (m >= A.nt) {
    k++;
    m = m - A.nt + k;
  }
  n = 0;

  double one = 1., none = -1.;

  unsigned long row = 0, col = 0;

  auto findSparePtr = [&]() {
    double *ptr;
    // NOTE: Jie: Only for lower!
    if (allocated >= maxBound) {
      while (ms_get_ptr(row, col) == nullptr) {
        col++;
        if (col > row) {
          row++;
          col = 0;
        }
      }
      while (ms_get_count(row, col) != 0) {
        plasma_yield();
      }
      ptr = ms_get_ptr(row, col);
      ms_get_ptr(row, col) = nullptr;
    } else {
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&ptr), bufferSizeInByte));
      allocated++;
    }
    return ptr;
  };

  while (k < A.nt && m < A.nt && !ss_aborted()) {
    next_n = n;
    next_m = m;
    next_k = k;

    next_n++;
    if (next_n > next_k) {
      next_m += PLASMA_SIZE;
      while (next_m >= A.nt && next_k < A.nt) {
        next_k++;
        next_m = next_m - A.nt + next_k;
      }
      next_n = 0;
    }

    tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
    tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

    ldak = BLKLDD(A, k);
    ldan = BLKLDD(A, n);
    ldam = BLKLDD(A, m);

    if (m == k) {
      if (n == k) {
        ms_get_count(k, k)++;
        if (k == 0) {
          double *ptr = findSparePtr();
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldak * sizeof(double), A(k, k), ldak * sizeof(double),
              tempkn * sizeof(double), tempkn, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(k, k) = ptr;
        }
      } else {
        ms_get_count(k, k)++;
        if (n == 0) {
          double *ptr = findSparePtr();
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldak * sizeof(double), A(k, k), ldak * sizeof(double),
              tempkn * sizeof(double), tempkn, cudaMemcpyHostToDevice,
              plasma->cuda_stream[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldak * sizeof(double) * tempkn;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(k, k) = ptr;
        }
        ms_get_count(k, n)++;
        if (ms_get_ptr(k, n) == nullptr) {
          double *ptr = findSparePtr();
          ss_cond_wait(k, n, 1);
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldak * sizeof(double), A(k, n), ldak * sizeof(double),
              tempkn * sizeof(double), A.nb, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(k, n) = ptr;
        }
      }
    } else {
      if (n == k) {
        ms_get_count(m, k)++;
        if (k == 0) {
          double *ptr = findSparePtr();
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldam * sizeof(double), A(m, k), ldam * sizeof(double),
              tempmn * sizeof(double), A.nb, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(m, k) = ptr;
        }
        ms_get_count(k, k)++;
        if (ms_get_ptr(k, k) == nullptr) {
          double *ptr = findSparePtr();
          ss_cond_wait(k, k, 1);
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldak * sizeof(double), A(k, k), ldak * sizeof(double),
              A.nb * sizeof(double), A.nb, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(k, k) = ptr;
        }
      } else {
        ms_get_count(m, k)++;
        if (n == 0) {
          double *ptr = findSparePtr();
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldam * sizeof(double), A(m, k), ldam * sizeof(double),
              tempmn * sizeof(double), A.nb, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(m, k) = ptr;
        }
        ms_get_count(k, n)++;
        if (ms_get_ptr(k, n) == nullptr) {
          double *ptr = findSparePtr();
          ss_cond_wait(k, n, 1);
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldak * sizeof(double), A(k, n), ldak * sizeof(double),
              A.nb * sizeof(double), A.nb, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldak * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(k, n) = ptr;
        }

        ms_get_count(m, n)++;
        if (ms_get_ptr(m, n) == nullptr) {
          double *ptr = findSparePtr();
          ss_cond_wait(m, n, 1);
          start = get_current_time();
          CHECK_CUDA(cudaMemcpy2DAsync(
              ptr, ldam * sizeof(double), A(m, n), ldam * sizeof(double),
              tempmn * sizeof(double), A.nb, cudaMemcpyHostToDevice,
              plasma->cuda_stream_c2g[rank]));
          make_atomic(plasma->volumeCPU2GPU) += ldam * sizeof(double) * A.nb;
          CHECK_CUDA(cudaStreamSynchronize(plasma->cuda_stream_c2g[rank]));
          end = get_current_time();
          log_event(rank, rankDev, EVENT_C2G, start, end);
          ms_get_ptr(m, n) = ptr;
        }
      }
    }
    n = next_n;
    m = next_m;
    k = next_k;
  }
  for (unsigned long i = 0; i < A.nt; ++i) {
    for (unsigned long j = 0; j <= i; ++j) {
      if (ms_get_ptr(i, j) == nullptr) {
        continue;
      } else {
        while (ms_get_count(i, j) != 0) {
          plasma_yield();
        }
        CHECK_CUDA(cudaFree(ms_get_ptr(i, j)));
      }
    }
  }
}

void plasma_pdpotrf_gpu_reuse_data_static_table_dist(plasma_context_t *plasma) {
  const int rank = PLASMA_RANK;
  const int rankDev = plasma->rankMapping[rank];

  static_assert(sizeof(std::atomic<decltype(plasma->volumeCPU2GPU)>) ==
                sizeof(decltype(plasma->volumeCPU2GPU)));
  static_assert(sizeof(std::atomic<decltype(plasma->volumeGPU2CPU)>) ==
                sizeof(decltype(plasma->volumeGPU2CPU)));
  PLASMA_enum uplo;
  PLASMA_desc A;
  PLASMA_sequence *sequence;
  PLASMA_request *request;

  plasma_unpack_args_4(uplo, A, sequence, request);
  if (uplo == PlasmaUpper) {
    printf("Upper is not supported yet\n");
    return;
  }
  if (rank == 0) {
    plasma->volumeCPU2GPU = 0;
    plasma->volumeGPU2CPU = 0;
    plasma->ss_ld = A.nt;
    plasma->ss_abort = 0;
  }
  if (sequence->status != PLASMA_SUCCESS) return;
  ss_init(A.nt, A.nt, 0);
  auto memoryStatus = new MemoryStatus[A.nt * A.nt];
  int current_step = 0;

  cudaSetDevice(rankDev);
  size_t workspaceInBytes;
  CHECK_CUSOLVER(cusolverDnPotrf_bufferSize(
      plasma->cusolver_handle[rank], plasma->cusolver_params[rank],
      CUBLAS_FILL_MODE_LOWER, A.nb, CUDA_R_64F, NULL, A.nb, CUDA_R_64F,
      &workspaceInBytes));
  void *pBuffer;
  CHECK_CUDA(
      cudaMalloc(&pBuffer, workspaceInBytes +
                               (sizeof(int) - workspaceInBytes % sizeof(int)) +
                               sizeof(int)));
  int *infoDevice = (int *)((std::ptrdiff_t)pBuffer + workspaceInBytes +
                            (sizeof(int) - workspaceInBytes % sizeof(int)));

  std::thread t_c2g{plasma_pdpotrf_gpu_reuse_data_static_table_c2g_dist, rank,
                    rankDev, memoryStatus, plasma};
  std::thread t_compute{
      plasma_pdpotrf_gpu_reuse_data_static_table_computation_dist,
      rank,
      rankDev,
      &current_step,
      pBuffer,
      workspaceInBytes,
      infoDevice,
      memoryStatus,
      plasma};
  std::thread t_g2c{plasma_pdpotrf_gpu_reuse_data_static_table_g2c_dist,
                    rank,
                    rankDev,
                    memoryStatus,
                    &current_step,
                    plasma};

  while (!t_c2g.joinable()) {
    plasma_yield();
  }
  t_c2g.join();
  while (!t_compute.joinable()) {
    plasma_yield();
  }
  t_compute.join();
  cudaFree(pBuffer);
  while (!t_g2c.joinable()) {
    plasma_yield();
  }
  t_g2c.join();

  delete[] memoryStatus;
  ss_finalize();
}

#endif

/***************************************************************************/
/**
 *  Parallel tile Cholesky factorization - dynamic scheduling
 **/

void plasma_pdpotrf_gpu_dist_quark(PLASMA_enum uplo, PLASMA_desc A,
                                   PLASMA_sequence *sequence,
                                   PLASMA_request *request) {
  printf("Dynamic scheduling is not supported yet\n");
  return;
}

void plasma_pdpotrf_gpu_reuse_data_dist_quark(PLASMA_enum uplo, PLASMA_desc A,
                                              PLASMA_sequence *sequence,
                                              PLASMA_request *request) {
  printf("Dynamic scheduling is not supported yet\n");
  return;
}

void plasma_pdpotrf_gpu_reuse_data_table_dist_quark(PLASMA_enum uplo,
                                                    PLASMA_desc A,
                                                    PLASMA_sequence *sequence,
                                                    PLASMA_request *request) {
  printf("Dynamic scheduling is not supported yet\n");
  return;
}

void plasma_pdpotrf_gpu_reuse_data_table_all_managed_dist_quark(
    PLASMA_enum uplo, PLASMA_desc A, PLASMA_sequence *sequence,
    PLASMA_request *request) {
  printf("Dynamic scheduling is not supported yet\n");
  return;
}

void plasma_pdpotrf_gpu_reuse_data_table_no_free_dist_quark(
    PLASMA_enum uplo, PLASMA_desc A, PLASMA_sequence *sequence,
    PLASMA_request *request) {
  printf("Dynamic scheduling is not supported yet\n");
  return;
}

void plasma_pdpotrf_gpu_reuse_data_static_table_dist_quark(
    PLASMA_enum uplo, PLASMA_desc A, PLASMA_sequence *sequence,
    PLASMA_request *request) {
  printf("Dynamic scheduling is not supported yet\n");
  return;
}
