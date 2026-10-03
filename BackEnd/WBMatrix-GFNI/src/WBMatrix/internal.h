#ifndef WBMATRIX2_INTERNAL_H
#define WBMATRIX2_INTERNAL_H

#include <WBMatrix/WBMatrix.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

typedef struct M16PreparedB_GPT {
    __m128i B_top_t;
    __m128i B_bot_t;
} M16PreparedB_GPT;

typedef struct M32PreparedB_GPT {
    __m256i BT[4];
} M32PreparedB_GPT;

typedef struct M64PreparedB_M32PairBT_GPT {
    __m512i B00_B01[4];
    __m512i B10_B11[4];
} M64PreparedB_M32PairBT_GPT;

typedef struct M128PreparedB_8x8Tiles_GPT {
    __m512i BT[16][2];
} M128PreparedB_8x8Tiles_GPT;

typedef struct M128TileCtx_GPT {
    __m512i transpose;
    __m512i vpermb_mask;
    __m512i pack_mask;
    __m512i idx_left;
    __m512i idx_right;
    __m512i idx_pack0;
    __m512i idx_pack1;
    __m512i bcast_lane[8];
} M128TileCtx_GPT;

extern unsigned int randseed;

void wbmatrix2_matmul_m8_two_gfni(M8 left, M8 right, M8 *out);
void wbmatrix2_matmul_m8_three_gfni(M8 left, M8 right, M8 *out);

/* Deterministic HBUP cores. The caller initializes the WBMatrix PRNG. */
void wbmatrix2_genmatpair_m4_hbup(M4 *mat, M4 *mat_inv);
void wbmatrix2_genmatpair_m8_hbup(M8 *mat, M8 *mat_inv);
void wbmatrix2_genmatpair_m16_hbup(M16 *mat, M16 *mat_inv);
void wbmatrix2_genmatpair_m32_hbup(M32 *mat, M32 *mat_inv);
void wbmatrix2_genmatpair_m64_hbup(M64 *mat, M64 *mat_inv);
void wbmatrix2_genmatpair_m128_hbup(M128 *mat, M128 *mat_inv);
void wbmatrix2_genmatpair_m256_hbup(M256 *mat, M256 *mat_inv);

#endif
