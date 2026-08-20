/**
 * CUDA copy-direction detection for the plain (non-emulation) distributed
 * FP64 Cholesky driver in compute/pdpotrf_gpu_dist.cpp.
 *
 * Copied verbatim out of that file, unchanged, into its own header --
 * just relocated, not rewritten. Deliberately separate from
 * compute/mx_tile_copy.h's deduceMemcpyKind (used by the emulation-path
 * driver, compute/pdpotrf_gpu_mixed_precision.cpp): despite the identical
 * name, the two use different pointer-attribute logic and different
 * fallback behavior, so they are NOT the same function and must not be
 * merged into one.
 */
#ifndef DIST_TILE_COPY_H
#define DIST_TILE_COPY_H

#include <cuda_runtime.h>

namespace {
inline cudaMemcpyKind deduceMemcpyKind(const void *dst, const void *src) {
  cudaPointerAttributes dstAttr{}, srcAttr{};
  auto dstOk = cudaPointerGetAttributes(&dstAttr, dst) == cudaSuccess;
  auto srcOk = cudaPointerGetAttributes(&srcAttr, src) == cudaSuccess;
  if (dstOk && srcOk) {
    if (dstAttr.type != cudaMemoryTypeHost && srcAttr.type != cudaMemoryTypeHost)
      return cudaMemcpyDeviceToDevice;
    if (dstAttr.type == cudaMemoryTypeHost && srcAttr.type != cudaMemoryTypeHost)
      return cudaMemcpyDeviceToHost;
    if (dstAttr.type != cudaMemoryTypeHost && srcAttr.type == cudaMemoryTypeHost)
      return cudaMemcpyHostToDevice;
  }
  return cudaMemcpyDefault;
}
}  // namespace

#endif  // DIST_TILE_COPY_H
