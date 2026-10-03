#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>


static inline uint64_t m64_reverse64(uint64_t x)
{
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}


static inline uint64_t m64_avx512_vpopcntdq_movmask_core(const M64 *Mat, uint64_t n)
{
    const __m512i x = _mm512_set1_epi64((long long)n);
    uint64_t raw = 0;

    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i cnt0 = _mm512_popcnt_epi64(_mm512_and_si512(rows0, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt0, 63)) << 0;

    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[8]);
    const __m512i cnt1 = _mm512_popcnt_epi64(_mm512_and_si512(rows1, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt1, 63)) << 8;

    const __m512i rows2 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt2 = _mm512_popcnt_epi64(_mm512_and_si512(rows2, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt2, 63)) << 16;

    const __m512i rows3 = _mm512_loadu_si512((const void *)&Mat->M[24]);
    const __m512i cnt3 = _mm512_popcnt_epi64(_mm512_and_si512(rows3, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt3, 63)) << 24;

    const __m512i rows4 = _mm512_loadu_si512((const void *)&Mat->M[32]);
    const __m512i cnt4 = _mm512_popcnt_epi64(_mm512_and_si512(rows4, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt4, 63)) << 32;

    const __m512i rows5 = _mm512_loadu_si512((const void *)&Mat->M[40]);
    const __m512i cnt5 = _mm512_popcnt_epi64(_mm512_and_si512(rows5, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt5, 63)) << 40;

    const __m512i rows6 = _mm512_loadu_si512((const void *)&Mat->M[48]);
    const __m512i cnt6 = _mm512_popcnt_epi64(_mm512_and_si512(rows6, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt6, 63)) << 48;

    const __m512i rows7 = _mm512_loadu_si512((const void *)&Mat->M[56]);
    const __m512i cnt7 = _mm512_popcnt_epi64(_mm512_and_si512(rows7, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt7, 63)) << 56;

    return m64_reverse64(raw);
}


uint64_t MatMulNumM64_avx512_vpopcntdq_movmask_ptr(const M64 *Mat, uint64_t n)
{
    return m64_avx512_vpopcntdq_movmask_core(Mat, n);
}

