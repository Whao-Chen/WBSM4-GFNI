#ifndef BAIWU_WBMATRIX_OPT_BACKEND_H
#define BAIWU_WBMATRIX_OPT_BACKEND_H

#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"

#define genMatpairM8(matrix, inverse) \
    genMatpairM8_radix2_unitriangular_plu_dynamicP_fastP_chain6_lowqword_invcols_sharedI_low8mask( \
        (matrix), (inverse))

#define genMatpairM32(matrix, inverse) \
    genMatpairM32_radix2_unitriangular_plu_dynamicP_recursiveBlocks_twoLeftTwoRight_rowdirectPinv( \
        (matrix), (inverse))

#define MatMulMatM32(left, right, output) \
    MatMulMatM32_rowdirect_ptr(&(left), &(right), (output))

#define MatMulNumM8(matrix, value) \
    MatMulNumM8_gfni_u64load_ptr(&(matrix), (value))

#define MatMulNumM32(matrix, value) \
    MatMulNumM32_avx512vl_vpopcntdq_ymm_ptr(&(matrix), (value))

#define affineU32(aff, value) \
    affineU32_matmulnum_gpt7_wrapper_ptr(&(aff), (value))

#endif
