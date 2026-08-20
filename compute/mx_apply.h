/**
 * MX/FP8 tile-application layer: given a tile of source data and a decided
 * target format name, actually materialize the quantized tile (allocate
 * pinned host storage, quantize via mx_emulation.h, and stamp the tile's
 * MX metadata). Used by both the Baseline heuristic path and the three
 * bound-selector ladder modes (see ladder_selection.h) to turn a format
 * decision into real tile data.
 *
 * Header-only: mappedBlock is a live Eigen expression template (a Block
 * view into the source matrix), so every function here is templated on
 * its concrete type rather than naming it explicitly.
 *
 * Relies on g_pinned_tiles/g_pinned_bytes (defined in
 * dpotrf_mixed_precision.cpp) for the same pinned-allocation bookkeeping
 * the rest of the tile-building code does.
 */
#ifndef MX_APPLY_H
#define MX_APPLY_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "cuda_fp8.h"

#include "mixed_precision.h"
#include "mx_emulation.h"

extern size_t g_pinned_tiles;
extern size_t g_pinned_bytes;

namespace mx_apply {

inline void log_tile_target(int row, int col, const char *target) {
  std::cout << "[TILE_TARGET] (" << row << ", " << col << ") "
            << target << std::endl;
}

// Normalize an aliased/legacy format-name spelling to the canonical one
// this codebase dispatches on; falls back to `fallback` if unrecognized.
inline std::string canonical_format_name(const std::string &raw,
                                         const char *fallback) {
  std::string v = raw.empty() ? std::string(fallback) : raw;
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "mx_f16") v = "mx_fp16";
  if (v == "mx_fp8_e4m3") v = "mx_e4m3";
  if (v == "mx_fp8_e5m2") v = "mx_e5m2";
  if (v == "fp8e4m3") v = "fp8_e4m3";
  if (v == "fp8e5m2") v = "fp8_e5m2";
  if (v == "fp6_e3m2" || v == "fp6e3m2") v = "e3m2";
  if (v == "fp6_e2m3" || v == "fp6e2m3") v = "e2m3";
  if (v == "fp4_e2m1" || v == "fp4e2m1") v = "e2m1";

  if (v == "fp32" || v == "fp16" ||
      v == "mx_fp16" ||
      v == "mx_e4m3" || v == "mx_e5m2" ||
      v == "fp8_e4m3" || v == "fp8_e5m2" ||
      v == "e3m2" || v == "e2m3" || v == "e2m1") {
    return v;
  }
  return std::string(fallback);
}

// MX block-scaled quantization: shared UE8M0 scale per kMxVecSize-element
// row-vector group, then per-element round to (ebits, mbits) -- or, for
// MXFP16 (ebits=5, mbits=10), a real IEEE binary16 cast instead of the
// custom quantizeFp grid, since MXFP16 just needs genuine FP16 rounding
// under the same block scale.
template <typename Derived>
void quant_mx_tile(MixedPrecisionTile &tile, int row, int col,
                    const Eigen::MatrixBase<Derived> &mappedBlock,
                    const char *label, int ebits, int mbits, float max_norm) {
  std::cout << "--- Tile (" << row << ", " << col
            << ") selected for CUDA_R_32F (" << label
            << " Quantization) ---" << std::endl;

  tile.dtype = CUDA_R_32F;
  tile.mx_ebits = ebits;
  tile.mx_mbits = mbits;
  tile.mx_max_norm = max_norm;
  tile.mx_scale_bits = 0;
  tile.mx_mode_block = 2;  // vec1d: 1D row-vector shared scale
  tile.mx_block_subtile = kMxVecSize;

  size_t tile_size = tile.m * tile.n;
  float *h_output = nullptr;

  cudaMallocHost((void **)&tile.data, sizeof(uint32_t) * tile_size);
  h_output = static_cast<float *>(tile.data);
  g_pinned_tiles++;
  g_pinned_bytes += sizeof(uint32_t) * tile_size;

  std::vector<float> tmp(tile_size);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
               Eigen::RowMajor>>{
    tmp.data(),
    static_cast<Eigen::Index>(tile.m),
    static_cast<Eigen::Index>(tile.n)} = mappedBlock.template cast<float>();

  const long total_elements = static_cast<long>(tile_size);
  float global_max = 0.0f;
  for (long i = 0; i < total_elements; ++i) {
    global_max = std::fmax(global_max, std::fabs(tmp[i]));
  }

  std::cout << "  Tile Size: " << total_elements
            << ", MaxVals: " << global_max
            << ", Mode: vec1d" << std::endl;

  const bool mx_fp16 = (ebits == 5 && mbits == 10);
  {
    // 1D row-vector: each row split into vectors of kMxVecSize elements, each with own scale
    const size_t vec_sz = static_cast<size_t>(kMxVecSize);
    const size_t tile_m = static_cast<size_t>(tile.m);
    const size_t tile_n = static_cast<size_t>(tile.n);
    for (size_t r = 0; r < tile_m; ++r) {
      float *rp = tmp.data() + r * tile_n;
      for (size_t c0 = 0; c0 < tile_n; c0 += vec_sz) {
        const size_t c_end = (c0 + vec_sz < tile_n) ? c0 + vec_sz : tile_n;
        float max_val = 0.0f;
        for (size_t c = c0; c < c_end; ++c)
          max_val = std::fmax(max_val, std::fabs(rp[c]));
        const int8_t vec_scale = computeScaleFromMax(max_val);
        const float scale = pow2(static_cast<int>(vec_scale));
        const float inv_scale = (scale == 0.0f) ? 1.0f : 1.0f / scale;
        for (size_t c = c0; c < c_end; ++c) {
          const float x = rp[c] * inv_scale;
          if (mx_fp16)
            rp[c] = static_cast<float>(static_cast<Eigen::half>(x)) * scale;
          else
            rp[c] = quantizeFp(x, ebits, mbits, max_norm) * scale;
        }
      }
    }
  }
  for (size_t i = 0; i < tile_size; ++i)
    h_output[i] = tmp[i];
}

// Plain FP8 quantization: no MX block scale at all, just element-wise
// round to (ebits, mbits) at the value's natural magnitude.
template <typename Derived>
void quant_plain_fp_tile(MixedPrecisionTile &tile, int row, int col,
                          const Eigen::MatrixBase<Derived> &mappedBlock,
                          const char *label, int ebits, int mbits, float max_norm) {
  std::cout << "--- Tile (" << row << ", " << col
            << ") selected for CUDA_R_32F (" << label
            << " Quantization) ---" << std::endl;

  tile.dtype = CUDA_R_32F;
  // mx_scale_bits = -1 signals plain FP8 (no MX block scale) to requantizeTileHost
  tile.mx_ebits = ebits;
  tile.mx_mbits = mbits;
  tile.mx_max_norm = max_norm;
  tile.mx_scale_bits = -1;
  tile.mx_mode_block = 0;
  tile.mx_block_subtile = 0;

  size_t tile_size = tile.m * tile.n;
  float *h_output = nullptr;

  cudaMallocHost((void **)&tile.data, sizeof(uint32_t) * tile_size);
  h_output = static_cast<float *>(tile.data);
  g_pinned_tiles++;
  g_pinned_bytes += sizeof(uint32_t) * tile_size;

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
               Eigen::RowMajor>>{
    h_output,
    static_cast<Eigen::Index>(tile.m),
    static_cast<Eigen::Index>(tile.n)} = mappedBlock.template cast<float>();

  const long total_elements = static_cast<long>(tile_size);
  for (long i = 0; i < total_elements; ++i) {
    h_output[i] = quantizeFp(h_output[i], ebits, mbits, max_norm);
  }
}

// Genuine native hardware FP8 (E4M3) storage -- requires compute
// capability >= 9 (Hopper+). Copied verbatim from the true original
// (pre-emulation) code path: writes the real __nv_fp8_e4m3 hardware type
// directly, no quantizeFp/FP32-container emulation involved at all. Use
// this when hasFp8 is true; quant_plain_fp_tile (above) is the emulated
// fallback for GPUs without native FP8 (e.g. Ampere/A100), which is what
// makes FP8-tier tiles usable/testable on hardware that doesn't actually
// have the format.
template <typename Derived>
void store_native_fp8_e4m3_tile(MixedPrecisionTile &tile,
                                const Eigen::MatrixBase<Derived> &mappedBlock) {
  tile.dtype = CUDA_R_8F_E4M3;
  cudaMallocHost(&tile.data, sizeof(uint8_t) * tile.m * tile.n);
  Eigen::Map<Eigen::Matrix<__nv_fp8_e4m3, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>{
      static_cast<__nv_fp8_e4m3 *>(tile.data),
      static_cast<Eigen::Index>(tile.m),
      static_cast<Eigen::Index>(tile.n)} =
      mappedBlock.template cast<__nv_fp8_e4m3>();
}

// Dispatch a canonical format-name string to the matching apply_* helper
// (fp32/fp16 stored verbatim; the rest quantized via quant_mx_tile or
// quant_plain_fp_tile). Returns false for an unrecognized name.
template <typename Derived>
bool dispatch_bucket_value(const std::string &v, MixedPrecisionTile &tile,
                            int row, int col,
                            const Eigen::MatrixBase<Derived> &mappedBlock) {
  if (v == "fp32") {
    log_tile_target(row, col, "fp32");
    tile.dtype = CUDA_R_32F;
    cudaMallocHost(&tile.data, sizeof(uint32_t) * tile.m * tile.n);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>{
        static_cast<float *>(tile.data),
        static_cast<Eigen::Index>(tile.m),
        static_cast<Eigen::Index>(tile.n)} = mappedBlock.template cast<float>();
    return true;
  }
  if (v == "fp16") {
    log_tile_target(row, col, "fp16");
    tile.dtype = CUDA_R_16F;
    cudaMallocHost(&tile.data, sizeof(uint16_t) * tile.m * tile.n);
    Eigen::Map<Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>{
        static_cast<Eigen::half *>(tile.data),
        static_cast<Eigen::Index>(tile.m),
        static_cast<Eigen::Index>(tile.n)} = mappedBlock.template cast<Eigen::half>();
    return true;
  }
  if (v == "mx_fp16") {
    log_tile_target(row, col, "mx_fp16");
    quant_mx_tile(tile, row, col, mappedBlock, "MX_FP16", 5, 10, 65504.0f);
    return true;
  }
  if (v == "mx_e4m3" || v == "e4m3") {
    log_tile_target(row, col, "mx_e4m3");
    quant_mx_tile(tile, row, col, mappedBlock, "MX_E4M3", 4, 3, 448.0f);
    return true;
  }
  if (v == "mx_e5m2" || v == "e5m2") {
    log_tile_target(row, col, "mx_e5m2");
    quant_mx_tile(tile, row, col, mappedBlock, "MX_E5M2", 5, 2, 57344.0f);
    return true;
  }
  if (v == "fp8_e4m3" || v == "fp8e4m3") {
    log_tile_target(row, col, "fp8_e4m3");
    quant_plain_fp_tile(tile, row, col, mappedBlock, "FP8_E4M3", 4, 3, 448.0f);
    return true;
  }
  if (v == "fp8_e5m2" || v == "fp8e5m2") {
    log_tile_target(row, col, "fp8_e5m2");
    quant_plain_fp_tile(tile, row, col, mappedBlock, "FP8_E5M2", 5, 2, 57344.0f);
    return true;
  }
  if (v == "e3m2") {
    log_tile_target(row, col, "e3m2");
    quant_mx_tile(tile, row, col, mappedBlock, "MX_E3M2", 3, 2, 28.0f);
    return true;
  }
  if (v == "e2m3") {
    log_tile_target(row, col, "e2m3");
    quant_mx_tile(tile, row, col, mappedBlock, "MX_E2M3", 2, 3, 7.5f);
    return true;
  }
  if (v == "e2m1") {
    log_tile_target(row, col, "e2m1");
    quant_mx_tile(tile, row, col, mappedBlock, "MX_E2M1", 2, 1, 6.0f);
    return true;
  }
  return false;
}

// Apply `bucket` (or `fallback` if empty/unrecognized) to the tile.
template <typename Derived>
void apply_bucket(const std::string &bucket, const char *fallback,
                   MixedPrecisionTile &tile, int row, int col,
                   const Eigen::MatrixBase<Derived> &mappedBlock) {
  std::string v = bucket.empty() ? std::string(fallback) : bucket;
  if (!dispatch_bucket_value(v, tile, row, col, mappedBlock)) {
    std::cout << "[WARN] Unknown bucket format '" << v
              << "', using " << fallback << std::endl;
    dispatch_bucket_value(fallback, tile, row, col, mappedBlock);
  }
}

// Re-apply quantization to an already-quantized tile after GPU arithmetic
// has touched it. Needed because emulated low-precision tiles are stored
// in genuine CUDA_R_32F (real FP32) containers, not actual narrow-bit
// formats -- see quant_mx_tile/quant_plain_fp_tile above, both of which
// set tile.dtype = CUDA_R_32F. Real FP32-precision GPU ops (TRSM) on that
// data produce values that generally don't land back on the format's
// quantization grid, so this re-snaps them, using the same ebits/mbits/
// scale metadata the tile was originally quantized with (tile->mx_*,
// stamped by quant_mx_tile/quant_plain_fp_tile at tiling time).
inline void requantize_tile_host(const MixedPrecisionTile *tile) {
  if (tile->mx_ebits == 0) return;
  if (tile->dtype != CUDA_R_32F) return;

  float *data = static_cast<float *>(tile->data);
  const size_t tile_size = tile->m * tile->n;
  const int ebits = tile->mx_ebits;
  const int mbits = tile->mx_mbits;
  const float max_norm = tile->mx_max_norm;
  const int scale_bits = tile->mx_scale_bits;
  const bool mx_fp16 = (ebits == 5 && mbits == 10);

  // Plain FP8: no MX block scaling, just clamp/round each element
  if (scale_bits < 0) {
    for (size_t i = 0; i < tile_size; ++i)
      data[i] = quantizeFp(data[i], ebits, mbits, max_norm);
    return;
  }

  // vec1d mode: each row split into groups of kMxVecSize elements, each with
  // its own shared scale (the only MX mode this build supports).
  const size_t vec_sz = static_cast<size_t>(kMxVecSize);
  for (size_t r = 0; r < tile->m; ++r) {
    float *rp = data + r * tile->n;
    for (size_t c0 = 0; c0 < tile->n; c0 += vec_sz) {
      const size_t c_end = (c0 + vec_sz < tile->n) ? c0 + vec_sz : tile->n;
      float max_val = 0.0f;
      for (size_t c = c0; c < c_end; ++c)
        max_val = std::fmax(max_val, std::fabs(rp[c]));
      const int sc = computeScaleBitsFromMax(max_val, scale_bits);
      const float scale = pow2(sc);
      const float inv_scale = (scale == 0.0f) ? 1.0f : 1.0f / scale;
      for (size_t c = c0; c < c_end; ++c) {
        const float x = rp[c] * inv_scale;
        if (mx_fp16)
          rp[c] = static_cast<float>(static_cast<Eigen::half>(x)) * scale;
        else
          rp[c] = quantizeFp(x, ebits, mbits, max_norm) * scale;
      }
    }
  }
}

}  // namespace mx_apply

#endif  // MX_APPLY_H
