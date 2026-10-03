#ifndef MASKED_MATRIX_BINDINGS_H
#define MASKED_MATRIX_BINDINGS_H
#include "WBMatrix/WBMatrix.h"
#ifdef WBSM4_GFNI_BACKEND
#include "WBMatrixOpt/WBMatrixOpt.h"
static uint8_t backend_affine_u8(const Aff8 *affine, uint8_t value)
{
    return affineU8_gfni_u64load_ptr(affine, value);
}

static uint32_t backend_affine_u32(const Aff32 *affine, uint32_t value)
{
    return affineU32_matmulnum_gpt7_wrapper_ptr(affine, value);
}

static uint8_t backend_matmul_num_m8(const M8 *matrix, uint8_t value)
{
    return MatMulNumM8_gfni_u64load_ptr(matrix, value);
}

static uint32_t backend_matmul_num_m32(const M32 *matrix, uint32_t value)
{
    return MatMulNumM32_avx512vl_vpopcntdq_ymm_ptr(matrix, value);
}

static void backend_matmul_mat_m32(const M32 *left,
                                   const M32 *right,
                                   M32 *result)
{
    MatMulMatM32_rowdirect_ptr(left, right, result);
}

static void backend_gen_matpair_m8(M8 *matrix, M8 *inverse)
{
    genMatpairM8_radix2_unitriangular_plu_dynamicP_fastP_chain6_lowqword_invcols_sharedI_low8mask(
        matrix, inverse);
}

static void backend_gen_affinepair_m8(Aff8 *affine, Aff8 *inverse)
{
    genaffinepairM8_radix2_finalgenmatpair_inline_cusrand_gfni(
        affine, inverse);
}

static void backend_gen_affinepair_m32(Aff32 *affine, Aff32 *inverse)
{
    genaffinepairM32_radix2_finalgenmatpair_inversevec_cusrand_bitreverse32(
        affine, inverse);
}
#define affineU8(a,x) backend_affine_u8(&(a),(x))
#define affineU32(a,x) backend_affine_u32(&(a),(x))
#define MatMulNumM8(a,x) backend_matmul_num_m8(&(a),(x))
#define MatMulNumM32(a,x) backend_matmul_num_m32(&(a),(x))
#define MatMulMatM32(a,b,r) backend_matmul_mat_m32(&(a),&(b),(r))
#define genMatpairM8(a,b) backend_gen_matpair_m8((a),(b))
#define genaffinepairM8(a,b) backend_gen_affinepair_m8((a),(b))
#define genaffinepairM32(a,b) backend_gen_affinepair_m32((a),(b))
#define affinemixM32(a,b,r) affinemixM32_final_matmulmat_matmulnum_wrapper_ptr(&(a),&(b),(r))
#endif
#endif
