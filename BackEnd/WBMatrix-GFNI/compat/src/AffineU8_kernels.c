#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>


static inline uint8_t affineu8_gfni_u64load_linear(const M8 *mat, uint8_t x)
{
    uint64_t rows;
    memcpy(&rows, mat->M, sizeof(rows));

    __m128i x_reg = _mm_cvtsi32_si128((int)x);
    __m128i mat_reg = _mm_cvtsi64_si128((long long)rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(x_reg, mat_reg, 0);

    return (uint8_t)_mm_cvtsi128_si32(out);
}

uint8_t affineU8_gfni_u64load(Aff8 aff, uint8_t arr)
{
    return (uint8_t)(affineu8_gfni_u64load_linear(&aff.Mat, arr) ^ aff.Vec.V);
}

uint8_t affineU8_gfni_u64load_ptr(const Aff8 *aff, uint8_t arr)
{
    return (uint8_t)(affineu8_gfni_u64load_linear(&aff->Mat, arr) ^ aff->Vec.V);
}

