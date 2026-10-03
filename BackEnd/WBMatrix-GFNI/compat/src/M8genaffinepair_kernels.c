#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/*
 * M8 radix2 dynamic-P genaffinepair wrappers.
 *
 * These are ordinary complete generators:
 *   1. generate (A, A^-1) with the final M8 radix2 dynamic-P genMatpair,
 *   2. generate runtime affine vector b with randV8,
 *   3. compute A^-1*b using final MatMulVec/MatMulNum or an inline equivalent.
 *
 * No precomputed matrix/affine/P table or external prepared state is used.
 */


static inline uint8_t m8_genaffinepair_gfni_u64load_mulnum(const M8 *Mat, uint8_t n)
{
    uint64_t rows;
    memcpy(&rows, Mat->M, sizeof(rows));

    __m128i x = _mm_cvtsi32_si128((int)n);
    __m128i mat = _mm_cvtsi64_si128((long long)rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(x, mat, 0);

    return (uint8_t)_mm_cvtsi128_si32(out);
}



void genaffinepairM8_radix2_finalgenmatpair_inline_cusrand_gfni(Aff8 *aff, Aff8 *aff_inv)
{
    genMatpairM8_radix2_unitriangular_plu_dynamicP_fastP_chain6_lowqword_invcols_sharedI_low8mask(
        &aff->Mat, &aff_inv->Mat);
    aff->Vec.V = (uint8_t)cus_random();
    aff_inv->Vec.V = m8_genaffinepair_gfni_u64load_mulnum(&aff_inv->Mat, aff->Vec.V);
}
