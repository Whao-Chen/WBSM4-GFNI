#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>


static inline uint32_t m32_genaffinepair_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

static inline uint32_t m32_genaffinepair_bitreverse32_mulnum(const M32 *Mat, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return m32_genaffinepair_reverse32(raw);
}


void genaffinepairM32_radix2_finalgenmatpair_inversevec_cusrand_bitreverse32(Aff32 *aff, Aff32 *aff_inv)
{
    genMatpairM32_radix2_unitriangular_plu_dynamicP_recursiveBlocks_twoLeftTwoRight_rowdirectPinv(
        &aff->Mat, &aff_inv->Mat);
    aff_inv->Vec.V = cus_random();
    aff->Vec.V = m32_genaffinepair_bitreverse32_mulnum(&aff->Mat, aff_inv->Vec.V);
}
