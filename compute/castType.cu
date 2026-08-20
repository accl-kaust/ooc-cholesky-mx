#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdio>

#include "mixed_precision.h"
#include <algorithm>
template <typename Src, typename Dst>
__global__ void castKernel(int m, int n, const void *source, int ldSource,
                           void *target, int ldTarget) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto workingRow = tid / n;
  auto workingCol = tid % n;
  if (workingRow >= m || workingCol >= n) {
    return;
  }
  static_cast<Dst *>(target)[workingCol + workingRow * ldTarget] =
      Dst(static_cast<const Src *>(source)[workingCol + workingRow * ldSource]);
}

void *castToBuffer(int m, int n, const void *source, cudaDataType sourceType,
                   int ldSource, void *target, cudaDataType targetType,
                   int ldTarget, cudaStream_t stream) {
  if (targetType == sourceType) {
    return const_cast<void *>(source);
  }
  dim3 block{std::min<unsigned int>(256, static_cast<unsigned int>(m * n))};
  dim3 grid{(m * n - 1) / block.x + 1};
  switch (sourceType) {
    case CUDA_R_64F: {
      switch (targetType) {
        case CUDA_R_32F:
          castKernel<double, float><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16F:
          castKernel<double, __half><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16BF:
          castKernel<double, __nv_bfloat16><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_8F_E4M3:
          castKernel<double, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_32F: {
      switch (targetType) {
        case CUDA_R_64F:
          castKernel<float, double><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16F:
          castKernel<float, __half><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16BF:
          castKernel<float, __nv_bfloat16><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_8F_E4M3:
          castKernel<float, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16F: {
      switch (targetType) {
        case CUDA_R_64F:
          castKernel<__half, double><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_32F:
          castKernel<__half, float><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16BF:
          castKernel<__half, __nv_bfloat16><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_8F_E4M3:
          castKernel<__half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_16BF: {
      switch (targetType) {
        case CUDA_R_64F:
          castKernel<__nv_bfloat16, double><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_32F:
          castKernel<__nv_bfloat16, float><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16F:
          castKernel<__nv_bfloat16, __half><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_8F_E4M3:
          castKernel<__nv_bfloat16, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        default:
          printf("Not supported combination at %s:%d\n", __FILE__, __LINE__);
          break;
      }
      break;
    }
    case CUDA_R_8F_E4M3: {
      switch (targetType) {
        case CUDA_R_64F:
          castKernel<__nv_fp8_e4m3, double><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_32F:
          castKernel<__nv_fp8_e4m3, float><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16F:
          castKernel<__nv_fp8_e4m3, __half><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
          break;
        case CUDA_R_16BF:
          castKernel<__nv_fp8_e4m3, __nv_bfloat16><<<grid, block, 0, stream>>>(
              m, n, source, ldSource, target, ldTarget);
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

  return target;
}
