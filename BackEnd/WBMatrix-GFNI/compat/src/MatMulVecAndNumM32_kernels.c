#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>


static inline uint32_t m32_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}


static inline uint32_t m32_avx512_vpopcntdq_bitreverse32_core(const M32 *Mat, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return m32_reverse32(raw);
}


uint32_t MatMulNumM32_avx512_vpopcntdq_bitreverse32_ptr(const M32 *Mat, uint32_t n)
{
    return m32_avx512_vpopcntdq_bitreverse32_core(Mat, n);
}


static inline uint32_t m32_avx512vl_vpopcntdq_ymm_core(const M32 *Mat, uint32_t n)
{
    const __m256i x = _mm256_set1_epi32((int)n);
    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i rows0 = _mm256_loadu_si256((const __m256i *)&Mat->M[0]);
    const __m256i rows1 = _mm256_loadu_si256((const __m256i *)&Mat->M[8]);
    const __m256i rows2 = _mm256_loadu_si256((const __m256i *)&Mat->M[16]);
    const __m256i rows3 = _mm256_loadu_si256((const __m256i *)&Mat->M[24]);
    const uint32_t raw = (uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows0, x)), ones) |
                         ((uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows1, x)), ones) << 8) |
                         ((uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows2, x)), ones) << 16) |
                         ((uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows3, x)), ones) << 24);

    return m32_reverse32(raw);
}


uint32_t MatMulNumM32_avx512vl_vpopcntdq_ymm_ptr(const M32 *Mat, uint32_t n)
{
    return m32_avx512vl_vpopcntdq_ymm_core(Mat, n);
}

