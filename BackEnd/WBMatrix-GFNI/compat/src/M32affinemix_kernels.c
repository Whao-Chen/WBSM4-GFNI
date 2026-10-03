#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>


void affinemixM32_final_matmulmat_matmulnum_wrapper_ptr(const Aff32 *aff, const Aff32 *preaff_inv,
                                                             Aff32 *mixaff)
{
    Aff32 out;
    MatMulMatM32_rowdirect_ptr(&aff->Mat, &preaff_inv->Mat, &out.Mat);
    out.Vec.V = MatMulNumM32_avx512_vpopcntdq_bitreverse32_ptr(&aff->Mat, preaff_inv->Vec.V) ^ aff->Vec.V;
    *mixaff = out;
}

static inline uint32_t affinemix_m32_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

static inline uint32_t affinemix_m32_bitreverse32_mulnum_from_rows(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_inserti64x4(_mm512_castsi256_si512(A0), A1, 1);
    const __m512i rows1 = _mm512_inserti64x4(_mm512_castsi256_si512(A2), A3, 1);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return affinemix_m32_reverse32(raw);
}

static inline void affinemix_m32_rowdirect_matmul_from_rows(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32 *Mat2, M32 *Mat)
{
    const __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    const __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    const __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    const __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_reg_ymms(A0, A1, A2, A3, B0, B1, B2, B3, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}

static inline void affinemix_m32_gfni_fused_core(const Aff32 *aff, const Aff32 *preaff_inv, Aff32 *mixaff)
{
    const __m256i A0 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[0]);
    const __m256i A1 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[8]);
    const __m256i A2 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[16]);
    const __m256i A3 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[24]);

    affinemix_m32_rowdirect_matmul_from_rows(A0, A1, A2, A3, &preaff_inv->Mat, &mixaff->Mat);
    mixaff->Vec.V = affinemix_m32_bitreverse32_mulnum_from_rows(A0, A1, A2, A3, preaff_inv->Vec.V) ^
                    aff->Vec.V;
}

void affinemixM32_rowdirect_fused_bitreverse32(Aff32 aff, Aff32 preaff_inv, Aff32 *mixaff)
{
    affinemix_m32_gfni_fused_core(&aff, &preaff_inv, mixaff);
}

