#pragma once
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MixedPrecisionTile {
  int dtype;
  void *data;
  int layout;
  size_t m, n, ld;
  // MX re-quantization parameters — all 0 means not MX-quantized
  int mx_ebits;
  int mx_mbits;
  float mx_max_norm;
  int mx_scale_bits;    // 0 = full int8 scale, >0 = limited scale bits
  int mx_mode_block;    // 0 = tile-wide scale, 1 = block/row-wise scale
  int mx_block_subtile; // 0 = full row scale, >0 = subtile size
};

struct MixedPrecisionTiledArray {
  MixedPrecisionTile *tiles;
  int nt;
  int uplo;
  int nb, mb;
  size_t m, n;
};

size_t getSizeofTileElement(int dtype);

int getComputeType(int dtype);

void requantizeTileHost(const MixedPrecisionTile *tile);

void *castToBuffer(int m, int n, const void *source, cudaDataType sourceType,
                   int ldSource, void *target, cudaDataType targetType,
                   int ldTarget, cudaStream_t stream);

#ifdef __cplusplus
}
#endif