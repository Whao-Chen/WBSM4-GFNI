#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>


static inline uint8_t m8_gfni_u64load_core(const M8 *Mat, uint8_t n)
{
    uint64_t rows;
    memcpy(&rows, Mat->M, sizeof(rows));

    __m128i x = _mm_cvtsi32_si128((int)n);
    __m128i mat = _mm_cvtsi64_si128((long long)rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(x, mat, 0);

    return (uint8_t)_mm_cvtsi128_si32(out);
}


uint8_t MatMulNumM8_gfni_u64load_ptr(const M8 *Mat, uint8_t n)
{
    return m8_gfni_u64load_core(Mat, n);
}

