#ifndef WBDL_WBMATRIXOPT_BACKEND_H
#define WBDL_WBMATRIXOPT_BACKEND_H

#include "WBMatrixOpt/WBMatrixOpt.h"

/* Ordinary production recommendations from the 2026-06-29 final summary. */
static inline void wbdl_opt_gen_affine8(Aff8 *aff, Aff8 *aff_inv)
{
    genaffinepairM8_radix2_finalgenmatpair_inline_cusrand_gfni(
        aff, aff_inv);
}

static inline void wbdl_opt_gen_affine32(Aff32 *aff, Aff32 *aff_inv)
{
    genaffinepairM32_radix2_finalgenmatpair_inversevec_cusrand_bitreverse32(
        aff, aff_inv);
}

static inline uint8_t wbdl_opt_affine_u8(const Aff8 *aff, uint8_t value)
{
    return affineU8_gfni_u64load(*aff, value);
}

static inline uint32_t wbdl_opt_affine_u32(const Aff32 *aff, uint32_t value)
{
    return affineU32_matmulnum_gpt7_wrapper_ptr(aff, value);
}

static inline uint32_t wbdl_opt_mat_mul_num32(const M32 *matrix, uint32_t value)
{
    return MatMulNumM32_avx512vl_vpopcntdq_ymm_ptr(matrix, value);
}

static inline void wbdl_opt_mat_mul_mat32(const M32 *left, const M32 *right,
                                           M32 *result)
{
    MatMulMatM32_rowdirect_ptr(left, right, result);
}

static inline void wbdl_opt_affine_mix32(const Aff32 *aff,
                                          const Aff32 *previous_inverse,
                                          Aff32 *result)
{
    affinemixM32_final_matmulmat_matmulnum_wrapper_ptr(
        aff, previous_inverse, result);
}

#endif
