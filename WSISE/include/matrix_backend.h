#ifndef WSISE_WBMATRIX_OPT_BACKEND_H
#define WSISE_WBMATRIX_OPT_BACKEND_H

#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"

#define genaffinepairM8(aff, aff_inv) \
    genaffinepairM8_radix2_finalgenmatpair_inline_cusrand_gfni( \
        (aff), (aff_inv))

#define genaffinepairM32(aff, aff_inv) \
    genaffinepairM32_radix2_finalgenmatpair_inversevec_cusrand_bitreverse32( \
        (aff), (aff_inv))

#define affinemixM32(aff, preaff_inv, mixaff) \
    affinemixM32_final_matmulmat_matmulnum_wrapper_ptr( \
        &(aff), &(preaff_inv), (mixaff))

#define affineU8(aff, value) \
    affineU8_gfni_u64load_ptr(&(aff), (value))

#define affineU32(aff, value) \
    affineU32_matmulnum_gpt7_wrapper_ptr(&(aff), (value))

#define MatMulNumM64(matrix, value) \
    MatMulNumM64_avx512_vpopcntdq_movmask_ptr(&(matrix), (value))

#endif
