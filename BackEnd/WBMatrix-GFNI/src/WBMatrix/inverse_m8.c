#include "internal.h"


static inline __m128i invsM8_gpt_mul_preBt_xmm(__m128i A_rows, __m128i B_t);
static inline __m128i invsM8_gpt_square_xmm(__m128i A_rows, __m128i I);
static inline __m128i invsM8_GPT12_pow_gfni_preA_unrolled_xmm_core(__m128i A);
static void invsM8_GPT12_pow_gfni_preA_unrolled_xmm_ptr(const M8 *Mat, M8 *Mat_inv);
void invsM8(M8 Mat, M8 *Mat_inv);

static inline __m128i invsM8_gpt_mul_preBt_xmm(__m128i A_rows, __m128i B_t)
{
    return _mm_gf2p8affine_epi64_epi8(A_rows, B_t, 0);
}

static inline __m128i invsM8_gpt_square_xmm(__m128i A_rows, __m128i I)
{
    __m128i A_t = _mm_gf2p8affine_epi64_epi8(I, A_rows, 0);

    return _mm_gf2p8affine_epi64_epi8(A_rows, A_t, 0);
}

static inline __m128i invsM8_GPT12_pow_gfni_preA_unrolled_xmm_core(__m128i A)
{
    const uint64_t I_transpose = 0x0102040810204080ULL;
    __m128i I = _mm_cvtsi64_si128((long long)I_transpose);
    __m128i A_t = _mm_gf2p8affine_epi64_epi8(I, A, 0);
    __m128i R = A;

#define INVS_M8_BIN_SQ_XMM() do { R = invsM8_gpt_square_xmm(R, I); } while (0)
#define INVS_M8_BIN_MULA_XMM() do { R = invsM8_gpt_mul_preBt_xmm(R, A_t); } while (0)

    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();
    INVS_M8_BIN_SQ_XMM(); INVS_M8_BIN_MULA_XMM();

#undef INVS_M8_BIN_SQ_XMM
#undef INVS_M8_BIN_MULA_XMM

    return R;
}

static void invsM8_GPT12_pow_gfni_preA_unrolled_xmm_ptr(const M8 *Mat, M8 *Mat_inv)
{
    __m128i A = _mm_loadl_epi64((const __m128i *)(const void *)Mat->M);
    __m128i Inv = invsM8_GPT12_pow_gfni_preA_unrolled_xmm_core(A);

    _mm_storel_epi64((__m128i *)(void *)Mat_inv->M, Inv);
}

void invsM8(M8 Mat, M8 *Mat_inv)
{
    invsM8_GPT12_pow_gfni_preA_unrolled_xmm_ptr(&Mat, Mat_inv);
}
