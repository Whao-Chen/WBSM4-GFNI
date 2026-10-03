#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>

static inline uint64_t gpt8_random64(void) {
    const uint64_t high = (uint64_t)cus_random();
    const uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}


static inline __m128i gpt8_make_dynamic_p_mask_fast_bitsliced_low8(void) {
    uint8_t p[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32_t r = cus_random();

    const uint32_t j7 = r & 7U; r >>= 3;
    uint8_t t = p[7]; p[7] = p[j7]; p[j7] = t;

    uint32_t j6 = r & 7U; r >>= 3;
    j6 -= (uint32_t)(j6 == 7U);
    t = p[6]; p[6] = p[j6]; p[j6] = t;

    uint32_t j5 = r & 7U; r >>= 3;
    j5 -= ((uint32_t)(j5 >= 6U) << 1);
    t = p[5]; p[5] = p[j5]; p[j5] = t;

    uint32_t j4 = r & 7U; r >>= 3;
    j4 -= (uint32_t)(j4 >= 5U) * 5U;
    t = p[4]; p[4] = p[j4]; p[j4] = t;

    const uint32_t j3 = r & 3U; r >>= 2;
    t = p[3]; p[3] = p[j3]; p[j3] = t;

    uint32_t j2 = r & 3U; r >>= 2;
    j2 -= (uint32_t)(j2 == 3U);
    t = p[2]; p[2] = p[j2]; p[j2] = t;

    const uint32_t j1 = r & 1U;
    t = p[1]; p[1] = p[j1]; p[j1] = t;

    return _mm_loadl_epi64((const __m128i *)p);
}


static inline void gpt8_gen_plu_base_no_p_invcols_chain6_sharedI(
    __m128i *out_base,
    __m128i *out_base_inv_cols,
    __m128i I)
{
    const uint64_t r = gpt8_random64();

    const uint64_t m1_val = 0x0102040810204080ULL
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = 0x0102040810204080ULL
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = 0x0102040810204080ULL
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    const __m128i M1 = _mm_cvtsi64_si128((long long)m1_val);
    const __m128i M2 = _mm_cvtsi64_si128((long long)m2_val);
    const __m128i M3 = _mm_cvtsi64_si128((long long)m3_val);

    const uint64_t r_high = r >> 28;
    const uint64_t n1_val = 0x0102040810204080ULL
        ^ (r_high & 0xFULL)
        ^ (((r_high >> 4) & 0xFULL) << 8)
        ^ (((r_high >> 8) & 0xFULL) << 16)
        ^ (((r_high >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = 0x0102040810204080ULL
        ^ (((r_high >> 16) & 0x3ULL) << 4)
        ^ (((r_high >> 18) & 0x3ULL) << 12)
        ^ (((r_high >> 20) & 0x3ULL) << 32)
        ^ (((r_high >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = 0x0102040810204080ULL
        ^ (((r_high >> 24) & 1ULL) << 6)
        ^ (((r_high >> 25) & 1ULL) << 20)
        ^ (((r_high >> 26) & 1ULL) << 34)
        ^ (((r_high >> 27) & 1ULL) << 48);

    const __m128i N1 = _mm_cvtsi64_si128((long long)n1_val);
    const __m128i N2 = _mm_cvtsi64_si128((long long)n2_val);
    const __m128i N3 = _mm_cvtsi64_si128((long long)n3_val);

    __m128i base_t = _mm_gf2p8affine_epi64_epi8(I, N3, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, N2, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, N1, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, M3, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, M2, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, M1, 0);
    *out_base = _mm_gf2p8affine_epi64_epi8(I, base_t, 0);

    __m128i inv_cols = _mm_gf2p8affine_epi64_epi8(I, M1, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, M2, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, M3, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, N1, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, N2, 0);
    *out_base_inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, N3, 0);
}


void genMatpairM8_radix2_unitriangular_plu_dynamicP_fastP_chain6_lowqword_invcols_sharedI_low8mask(M8 *Mat, M8 *Mat_inv) {
    const uint64_t I_transpose = 0x0102040810204080ULL;
    const __m128i I = _mm_cvtsi64_si128((long long)I_transpose);
    __m128i mat_reg;
    __m128i inv_cols;
    gpt8_gen_plu_base_no_p_invcols_chain6_sharedI(&mat_reg, &inv_cols, I);

    const __m128i p_mask = gpt8_make_dynamic_p_mask_fast_bitsliced_low8();
    mat_reg = _mm_shuffle_epi8(mat_reg, p_mask);
    inv_cols = _mm_shuffle_epi8(inv_cols, p_mask);
    const __m128i inv_reg = _mm_gf2p8affine_epi64_epi8(I, inv_cols, 0);

    _mm_storel_epi64((__m128i *)Mat->M, mat_reg);
    _mm_storel_epi64((__m128i *)Mat_inv->M, inv_reg);
}

