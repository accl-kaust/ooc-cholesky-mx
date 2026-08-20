#include <casting.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <thrust/transform.h>
#include <thrust/async/transform.h>

namespace mixed_kernels {
namespace {
template <typename From, typename To>
void casting1D(const cudaStream_t &stream, const void *from, void *to,
               size_t size) {
  // thrust::cuda::par_nosync.on makes sure async running
  thrust::transform(
      thrust::cuda::par_nosync.on(stream), static_cast<const From *>(from),
      &static_cast<const From *>(from)[size], static_cast<To *>(to),
      [] __device__(const From &from) -> To { return static_cast<To>(from); });
}

template <typename From, typename To>
__global__ void casting2DRowMajorKernel(int m, int n, const void *from,
                                        size_t ldFrom, void *to, size_t ldTo) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto rowId = tid / n;
  auto colId = tid % n;
  if (rowId >= m || colId >= n) {
    return;
  }
  reinterpret_cast<To *>(static_cast<uint8_t *>(to) + rowId * ldTo)[colId] =
      static_cast<To>(reinterpret_cast<const From *>(
          static_cast<const uint8_t*>(from) + rowId * ldFrom)[colId]);
}

template <typename From, typename To>
void casting2DRowMajor(const cudaStream_t &stream, int m, int n,
                       const void *from, size_t ldFrom, void *to, size_t ldTo) {
  std::size_t size = static_cast<std::size_t>(m) * n;
  dim3 block(std::min<std::size_t>(size, 256));
  dim3 grid((size - 1) / block.x + 1);
  casting2DRowMajorKernel<From, To>
      <<<grid, block, 0, stream>>>(m, n, from, ldFrom, to, ldTo);
}

template <typename From, typename To>
__global__ void plus2DRowMajorKernel(int m, int n, double beta,
                                     const void *from, size_t ldFrom, void *to,
                                     size_t ldTo) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto rowId = tid / n;
  auto colId = tid % n;
  if (rowId >= m || colId >= n) {
    return;
  }
  auto &fromElement = reinterpret_cast<const From *>(
      static_cast<const uint8_t *>(from) + rowId * ldFrom)[colId];
  auto &toElement = reinterpret_cast<To *>(static_cast<uint8_t *>(to) +
                                           rowId * ldTo)[colId];
  toElement = static_cast<To>(static_cast<double>(fromElement) +
                              beta * static_cast<double>(toElement));
}

template <typename From, typename To>
void plus2DRowMajor(const cudaStream_t &stream, int m, int n, double beta,
                    const void *from, size_t ldFrom, void *to, size_t ldTo) {
  std::size_t size = static_cast<std::size_t>(m) * n;
  dim3 block(std::min<std::size_t>(size, 256));
  dim3 grid((size - 1) / block.x + 1);
  plus2DRowMajorKernel<From, To>
      <<<grid, block, 0, stream>>>(m, n, beta, from, ldFrom, to, ldTo);
}
}  // namespace

const void *casting1D(const cudaStream_t &stream, const void *from,
                      cudaDataType fromType, void *to, cudaDataType toType,
                      size_t size) {
  if (toType == fromType) {
    return from;
  }
  switch (fromType) {
    case CUDA_R_64F: {
      switch (toType) {
        case CUDA_R_32F:
          casting1D<double, float>(stream, from, to, size);
          break;
        case CUDA_R_16F:
          casting1D<double, nv_half>(stream, from, to, size);
          break;
        case CUDA_R_16BF:
          casting1D<double, nv_bfloat16>(stream, from, to, size);
          break;
        case CUDA_R_8F_E4M3:
          casting1D<double, __nv_fp8_e4m3>(stream, from, to, size);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_32F: {
      switch (toType) {
        case CUDA_R_64F:
          casting1D<float, double>(stream, from, to, size);
          break;
        case CUDA_R_16F:
          casting1D<float, nv_half>(stream, from, to, size);
          break;
        case CUDA_R_16BF:
          casting1D<float, nv_bfloat16>(stream, from, to, size);
          break;
        case CUDA_R_8F_E4M3:
          casting1D<float, __nv_fp8_e4m3>(stream, from, to, size);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16F: {
      switch (toType) {
        case CUDA_R_64F:
          casting1D<nv_half, double>(stream, from, to, size);
          break;
        case CUDA_R_32F:
          casting1D<nv_half, float>(stream, from, to, size);
          break;
        case CUDA_R_16BF:
          casting1D<nv_half, nv_bfloat16>(stream, from, to, size);
          break;
        case CUDA_R_8F_E4M3:
          casting1D<nv_half, __nv_fp8_e4m3>(stream, from, to, size);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16BF: {
      switch (toType) {
        case CUDA_R_64F:
          casting1D<nv_bfloat16, double>(stream, from, to, size);
          break;
        case CUDA_R_32F:
          casting1D<nv_bfloat16, float>(stream, from, to, size);
          break;
        case CUDA_R_16F:
          casting1D<nv_bfloat16, nv_half>(stream, from, to, size);
          break;
        case CUDA_R_8F_E4M3:
          casting1D<nv_bfloat16, __nv_fp8_e4m3>(stream, from, to, size);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_8F_E4M3: {
      switch (toType) {
        case CUDA_R_64F:
          casting1D<__nv_fp8_e4m3, double>(stream, from, to, size);
          break;
        case CUDA_R_32F:
          casting1D<__nv_fp8_e4m3, float>(stream, from, to, size);
          break;
        case CUDA_R_16F:
          casting1D<__nv_fp8_e4m3, nv_half>(stream, from, to, size);
          break;
        case CUDA_R_16BF:
          casting1D<__nv_fp8_e4m3, nv_bfloat16>(stream, from, to, size);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    default:
      printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
      break;
  }

  return to;
}

const void *casting2DRowMajor(const cudaStream_t &stream, int m, int n,
                              const void *from, cudaDataType fromType,
                              size_t ldFrom, void *to, cudaDataType toType,
                              size_t ldTo) {
  if (toType == fromType) {
    return from;
  }
  switch (fromType) {
    case CUDA_R_64F: {
      switch (toType) {
        case CUDA_R_32F:
          casting2DRowMajor<double, float>(stream, m, n, from, ldFrom, to,
                                           ldTo);
          break;
        case CUDA_R_16F:
          casting2DRowMajor<double, nv_half>(stream, m, n, from, ldFrom, to,
                                             ldTo);
          break;
        case CUDA_R_16BF:
          casting2DRowMajor<double, nv_bfloat16>(stream, m, n, from, ldFrom, to,
                                                 ldTo);
          break;
        case CUDA_R_8F_E4M3:
          casting2DRowMajor<double, __nv_fp8_e4m3>(stream, m, n, from, ldFrom,
                                                   to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_32F: {
      switch (toType) {
        case CUDA_R_64F:
          casting2DRowMajor<float, double>(stream, m, n, from, ldFrom, to,
                                           ldTo);
          break;
        case CUDA_R_16F:
          casting2DRowMajor<float, nv_half>(stream, m, n, from, ldFrom, to,
                                            ldTo);
          break;
        case CUDA_R_16BF:
          casting2DRowMajor<float, nv_bfloat16>(stream, m, n, from, ldFrom, to,
                                                ldTo);
          break;
        case CUDA_R_8F_E4M3:
          casting2DRowMajor<float, __nv_fp8_e4m3>(stream, m, n, from, ldFrom,
                                                  to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16F: {
      switch (toType) {
        case CUDA_R_64F:
          casting2DRowMajor<nv_half, double>(stream, m, n, from, ldFrom, to,
                                             ldTo);
          break;
        case CUDA_R_32F:
          casting2DRowMajor<nv_half, float>(stream, m, n, from, ldFrom, to,
                                            ldTo);
          break;
        case CUDA_R_16BF:
          casting2DRowMajor<nv_half, nv_bfloat16>(stream, m, n, from, ldFrom,
                                                  to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          casting2DRowMajor<nv_half, __nv_fp8_e4m3>(stream, m, n, from, ldFrom,
                                                    to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16BF: {
      switch (toType) {
        case CUDA_R_64F:
          casting2DRowMajor<nv_bfloat16, double>(stream, m, n, from, ldFrom, to,
                                                 ldTo);
          break;
        case CUDA_R_32F:
          casting2DRowMajor<nv_bfloat16, float>(stream, m, n, from, ldFrom, to,
                                                ldTo);
          break;
        case CUDA_R_16F:
          casting2DRowMajor<nv_bfloat16, nv_half>(stream, m, n, from, ldFrom,
                                                  to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          casting2DRowMajor<nv_bfloat16, __nv_fp8_e4m3>(stream, m, n, from,
                                                        ldFrom, to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_8F_E4M3: {
      switch (toType) {
        case CUDA_R_64F:
          casting2DRowMajor<__nv_fp8_e4m3, double>(stream, m, n, from, ldFrom,
                                                   to, ldTo);
          break;
        case CUDA_R_32F:
          casting2DRowMajor<__nv_fp8_e4m3, float>(stream, m, n, from, ldFrom,
                                                  to, ldTo);
          break;
        case CUDA_R_16F:
          casting2DRowMajor<__nv_fp8_e4m3, nv_half>(stream, m, n, from, ldFrom,
                                                    to, ldTo);
          break;
        case CUDA_R_16BF:
          casting2DRowMajor<__nv_fp8_e4m3, nv_bfloat16>(stream, m, n, from,
                                                        ldFrom, to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    default:
      printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
      break;
  }

  return to;
}

void plus2DRowMajor(const cudaStream_t &stream, int m, int n, double beta,
                    const void *from, cudaDataType fromType, size_t ldFrom,
                    void *to, cudaDataType toType, size_t ldTo) {
  switch (fromType) {
    case CUDA_R_64F: {
      switch (toType) {
        case CUDA_R_64F:
          plus2DRowMajor<double, double>(stream, m, n, beta, from, ldFrom, to,
                                         ldTo);
          break;
        case CUDA_R_32F:
          plus2DRowMajor<double, float>(stream, m, n, beta, from, ldFrom, to,
                                        ldTo);
          break;
        case CUDA_R_16F:
          plus2DRowMajor<double, nv_half>(stream, m, n, beta, from, ldFrom, to,
                                          ldTo);
          break;
        case CUDA_R_16BF:
          plus2DRowMajor<double, nv_bfloat16>(stream, m, n, beta, from, ldFrom,
                                              to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          plus2DRowMajor<double, __nv_fp8_e4m3>(stream, m, n, beta, from,
                                                ldFrom, to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_32F: {
      switch (toType) {
        case CUDA_R_64F:
          plus2DRowMajor<float, double>(stream, m, n, beta, from, ldFrom, to,
                                        ldTo);
          break;
        case CUDA_R_32F:
          plus2DRowMajor<float, float>(stream, m, n, beta, from, ldFrom, to,
                                       ldTo);
          break;
        case CUDA_R_16F:
          plus2DRowMajor<float, nv_half>(stream, m, n, beta, from, ldFrom, to,
                                         ldTo);
          break;
        case CUDA_R_16BF:
          plus2DRowMajor<float, nv_bfloat16>(stream, m, n, beta, from, ldFrom,
                                             to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          plus2DRowMajor<float, __nv_fp8_e4m3>(stream, m, n, beta, from, ldFrom,
                                               to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16F: {
      switch (toType) {
        case CUDA_R_64F:
          plus2DRowMajor<nv_half, double>(stream, m, n, beta, from, ldFrom, to,
                                          ldTo);
          break;
        case CUDA_R_32F:
          plus2DRowMajor<nv_half, float>(stream, m, n, beta, from, ldFrom, to,
                                         ldTo);
          break;
        case CUDA_R_16F:
          plus2DRowMajor<nv_half, nv_half>(stream, m, n, beta, from, ldFrom, to,
                                           ldTo);
          break;
        case CUDA_R_16BF:
          plus2DRowMajor<nv_half, nv_bfloat16>(stream, m, n, beta, from, ldFrom,
                                               to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          plus2DRowMajor<nv_half, __nv_fp8_e4m3>(stream, m, n, beta, from,
                                                 ldFrom, to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16BF: {
      switch (toType) {
        case CUDA_R_64F:
          plus2DRowMajor<nv_bfloat16, double>(stream, m, n, beta, from, ldFrom,
                                              to, ldTo);
          break;
        case CUDA_R_32F:
          plus2DRowMajor<nv_bfloat16, float>(stream, m, n, beta, from, ldFrom,
                                             to, ldTo);
          break;
        case CUDA_R_16F:
          plus2DRowMajor<nv_bfloat16, nv_half>(stream, m, n, beta, from, ldFrom,
                                               to, ldTo);
          break;
        case CUDA_R_16BF:
          plus2DRowMajor<nv_bfloat16, nv_bfloat16>(stream, m, n, beta, from,
                                                   ldFrom, to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          plus2DRowMajor<nv_bfloat16, __nv_fp8_e4m3>(stream, m, n, beta, from,
                                                     ldFrom, to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_8F_E4M3: {
      switch (toType) {
        case CUDA_R_64F:
          plus2DRowMajor<__nv_fp8_e4m3, double>(stream, m, n, beta, from,
                                                ldFrom, to, ldTo);
          break;
        case CUDA_R_32F:
          plus2DRowMajor<__nv_fp8_e4m3, float>(stream, m, n, beta, from, ldFrom,
                                               to, ldTo);
          break;
        case CUDA_R_16F:
          plus2DRowMajor<__nv_fp8_e4m3, nv_half>(stream, m, n, beta, from,
                                                 ldFrom, to, ldTo);
          break;
        case CUDA_R_16BF:
          plus2DRowMajor<__nv_fp8_e4m3, nv_bfloat16>(stream, m, n, beta, from,
                                                     ldFrom, to, ldTo);
          break;
        case CUDA_R_8F_E4M3:
          plus2DRowMajor<__nv_fp8_e4m3, __nv_fp8_e4m3>(stream, m, n, beta, from,
                                                       ldFrom, to, ldTo);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    default:
      printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
      break;
  }
}

bool operator<(const cudaDataType_t lhs, const cudaDataType_t rhs) {
  switch (lhs) {
    case CUDA_R_64F:
      switch (rhs) {
        case CUDA_R_64F:
        case CUDA_R_32F:
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
          return false;
        default:
          printf("Not supported type at %s:%d\n", __FILE__, __LINE__);
          return true;
      }
    case CUDA_R_32F:
      switch (rhs) {
        case CUDA_R_64F:
          return true;
        case CUDA_R_32F:
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
          return false;
        default:
          printf("Not supported type at %s:%d\n", __FILE__, __LINE__);
          return true;
      }
    case CUDA_R_16F:
    case CUDA_R_16BF:
      switch (rhs) {
        case CUDA_R_64F:
        case CUDA_R_32F:
          return true;
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
          return false;
        default:
          printf("Not supported type at %s:%d\n", __FILE__, __LINE__);
          return true;
      }
    case CUDA_R_8F_E4M3:
      switch (rhs) {
        case CUDA_R_64F:
        case CUDA_R_32F:
        case CUDA_R_16F:
        case CUDA_R_16BF:
          return true;
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
          return false;
        default:
          printf("Not supported type at %s:%d\n", __FILE__, __LINE__);
          return true;
      }
    default:
      printf("Not supported type at %s:%d\n", __FILE__, __LINE__);
      return true;
  }
}

size_t cudaDataSize(cudaDataType type) {
  switch (type) {
    case CUDA_R_64F:
      return sizeof(double);
    case CUDA_R_32F:
      return sizeof(float);
    case CUDA_R_16BF:
      return sizeof(nv_bfloat16);
    case CUDA_R_16F:
      return sizeof(nv_half);
    case CUDA_R_8F_E4M3:
      return sizeof(__nv_fp8_e4m3);
    default:
      return 0;
  }
}
}  // namespace mixed_kernels
