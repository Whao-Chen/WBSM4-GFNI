#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>


static inline void MatMulMatM32_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
    const __m256i I = _mm256_set1_epi64x(0x0102040810204080ULL);

    const __m256i shuf_B = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i perm_B = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    __m256i BT0 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B0, shuf_B), perm_B), 0);
    __m256i BT1 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B1, shuf_B), perm_B), 0);
    __m256i BT2 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B2, shuf_B), perm_B), 0);
    __m256i BT3 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B3, shuf_B), perm_B), 0);

    const __m256i m0 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m1 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m2 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m3 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i perm_ctrl = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);

#define GPT1_CALC_M32_ROW(A_ROW, C_OUT) do { \
        __m256i aik0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m256i t0 = _mm256_gf2p8affine_epi64_epi8(BT0, aik0, 0); \
        __m256i t1 = _mm256_gf2p8affine_epi64_epi8(BT1, aik1, 0); \
        __m256i t2 = _mm256_gf2p8affine_epi64_epi8(BT2, aik2, 0); \
        __m256i t3 = _mm256_gf2p8affine_epi64_epi8(BT3, aik3, 0); \
        __m256i acc01 = _mm256_xor_si256(t0, t1); \
        __m256i acc23 = _mm256_xor_si256(t2, t3); \
        __m256i acc = _mm256_xor_si256(acc01, acc23); \
        (C_OUT) = _mm256_gf2p8affine_epi64_epi8(I, acc, 0); \
    } while (0)

    GPT1_CALC_M32_ROW(A0, *C0);
    GPT1_CALC_M32_ROW(A1, *C1);
    GPT1_CALC_M32_ROW(A2, *C2);
    GPT1_CALC_M32_ROW(A3, *C3);

#undef GPT1_CALC_M32_ROW

    const __m256i inv_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i inv_shuf = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    *C0 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C0, inv_perm), inv_shuf);
    *C1 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C1, inv_perm), inv_shuf);
    *C2 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C2, inv_perm), inv_shuf);
    *C3 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C3, inv_perm), inv_shuf);
}


/* 把两个 YMM 拼成一个 ZMM，低 256 bit 为 lo，高 256 bit 为 hi。 */


static inline void MatMulMatM32_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
    const __m256i I = _mm256_set1_epi64x(0x0102040810204080ULL);

    const __m256i shuf_B = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i perm_B = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    __m256i BT0 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B0, shuf_B), perm_B), 0);
    __m256i BT1 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B1, shuf_B), perm_B), 0);
    __m256i BT2 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B2, shuf_B), perm_B), 0);
    __m256i BT3 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B3, shuf_B), perm_B), 0);

    const __m256i m0 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m1 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m2 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m3 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i perm_ctrl = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);

#define GPT4_CALC_M32_ROW(A_ROW, C_OUT) do { \
        __m256i aik0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m256i t0 = _mm256_gf2p8affine_epi64_epi8(aik0, BT0, 0); \
        __m256i t1 = _mm256_gf2p8affine_epi64_epi8(aik1, BT1, 0); \
        __m256i t2 = _mm256_gf2p8affine_epi64_epi8(aik2, BT2, 0); \
        __m256i t3 = _mm256_gf2p8affine_epi64_epi8(aik3, BT3, 0); \
        __m256i acc01 = _mm256_xor_si256(t0, t1); \
        __m256i acc23 = _mm256_xor_si256(t2, t3); \
        (C_OUT) = _mm256_xor_si256(acc01, acc23); \
    } while (0)

    GPT4_CALC_M32_ROW(A0, *C0);
    GPT4_CALC_M32_ROW(A1, *C1);
    GPT4_CALC_M32_ROW(A2, *C2);
    GPT4_CALC_M32_ROW(A3, *C3);

#undef GPT4_CALC_M32_ROW

    const __m256i inv_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i inv_shuf = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    *C0 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C0, inv_perm), inv_shuf);
    *C1 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C1, inv_perm), inv_shuf);
    *C2 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C2, inv_perm), inv_shuf);
    *C3 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C3, inv_perm), inv_shuf);
}


void MatMulMatM32_rowdirect_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat)
{
    __m256i A0 = _mm256_loadu_si256((const __m256i *)&Mat1->M[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i *)&Mat1->M[8]);
    __m256i A2 = _mm256_loadu_si256((const __m256i *)&Mat1->M[16]);
    __m256i A3 = _mm256_loadu_si256((const __m256i *)&Mat1->M[24]);
    __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_rowdirect_core(A0, A1, A2, A3, B0, B1, B2, B3, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}


void MatMulMatM32_reg_ymms(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
    MatMulMatM32_rowdirect_core(A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3);
}


void MatMulMatM32_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat) {
    __m256i A0 = _mm256_loadu_si256((const __m256i *)&Mat1->M[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i *)&Mat1->M[8]);
    __m256i A2 = _mm256_loadu_si256((const __m256i *)&Mat1->M[16]);
    __m256i A3 = _mm256_loadu_si256((const __m256i *)&Mat1->M[24]);
    __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_core(A0, A1, A2, A3, B0, B1, B2, B3, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}

