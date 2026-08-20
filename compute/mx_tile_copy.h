/**
 * CUDA copy-direction detection for the mixed-precision (emulation) tile
 * transfer path: given a source and destination pointer, work out whether
 * a cudaMemcpy*Async call should be H2D/D2H/D2D/H2H by querying each
 * pointer's actual memory space, instead of assuming a fixed direction.
 *
 * Needed because emulation-path tiles (see mx_apply.h) aren't always
 * plain host or plain device allocations -- their backing storage can be
 * pinned host, device, or managed (unified) memory depending on how the
 * tile was produced, so the transfer direction has to be resolved at
 * runtime per call rather than hardcoded.
 *
 * Used exclusively by compute/pdpotrf_gpu_mixed_precision.cpp today.
 * Deliberately NOT shared with compute/pdpotrf_gpu_dist.cpp's own
 * deduceMemcpyKind (the plain, non-emulation FP64 driver) -- that one
 * uses different pointer-attribute logic, and unifying the two would
 * change one file's behavior, not just deduplicate it.
 */
#ifndef MX_TILE_COPY_H
#define MX_TILE_COPY_H

#include <cuda_runtime.h>

inline cudaMemcpyKind deduceMemcpyKind(const void *dst, const void *src) {
  cudaPointerAttributes dstAttr{};
  cudaPointerAttributes srcAttr{};
  auto dstStatus = cudaPointerGetAttributes(&dstAttr, dst);
  auto srcStatus = cudaPointerGetAttributes(&srcAttr, src);

  const bool dstDevice =
      dstStatus == cudaSuccess &&
      (dstAttr.type == cudaMemoryTypeDevice || dstAttr.type == cudaMemoryTypeManaged);
  const bool srcDevice =
      srcStatus == cudaSuccess &&
      (srcAttr.type == cudaMemoryTypeDevice || srcAttr.type == cudaMemoryTypeManaged);

  if (dstDevice && srcDevice) return cudaMemcpyDeviceToDevice;
  if (dstDevice && !srcDevice) return cudaMemcpyHostToDevice;
  if (!dstDevice && srcDevice) return cudaMemcpyDeviceToHost;
  return cudaMemcpyHostToHost;
}

#endif  // MX_TILE_COPY_H
