/*
 * Copyright (c) 2021, SCNUCrypto, South China Normal University
 * 
 * Apache License 2.0
 *
 */
#ifndef WBSM4_COMPAT_BACKEND_H
#define WBSM4_COMPAT_BACKEND_H
#include "WBMatrix/WBMatrix.h"
#include <x86intrin.h>
typedef struct M16PreparedB_GPT {
    __m128i B_top_t;
    __m128i B_bot_t;
} M16PreparedB_GPT;
uint8_t affineU8_gfni_u64load(Aff8 aff, uint8_t arr);
uint8_t affineU8_gfni_u64load_ptr(const Aff8 *aff, uint8_t arr);
void genMatpairM8_radix2_unitriangular_plu_dynamicP_fastP_chain6_lowqword_invcols_sharedI_low8mask(M8 *Mat, M8 *Mat_inv);
void genaffinepairM8_radix2_finalgenmatpair_inline_cusrand_gfni(Aff8 *aff, Aff8 *aff_inv);
uint8_t MatMulNumM8_gfni_u64load_ptr(const M8 *Mat, uint8_t n);
void MatMulMatM16_reg(__m128i A0, __m128i A1, __m128i B0, __m128i B1, __m128i *C0, __m128i *C1);
uint32_t affineU32_matmulnum_gpt7_wrapper_ptr(const Aff32 *aff, uint32_t arr);
uint32_t MatMulNumM32_avx512_vpopcntdq_bitreverse32_ptr(const M32 *Mat, uint32_t n);
uint32_t MatMulNumM32_avx512vl_vpopcntdq_ymm_ptr(const M32 *Mat, uint32_t n);
void genMatpairM32_radix2_unitriangular_plu_dynamicP_recursiveBlocks_twoLeftTwoRight_rowdirectPinv(M32 *Mat, M32 *Mat_inv);
void genaffinepairM32_radix2_finalgenmatpair_inversevec_cusrand_bitreverse32(Aff32 *aff, Aff32 *aff_inv);
void affinemixM32_final_matmulmat_matmulnum_wrapper_ptr(const Aff32 *aff, const Aff32 *preaff_inv,
                                                             Aff32 *mixaff);
void affinemixM32_rowdirect_fused_bitreverse32(Aff32 aff, Aff32 preaff_inv, Aff32 *mixaff);
void MatMulMatM32_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat);
void MatMulMatM32_reg_ymms(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
void MatMulMatM32_rowdirect_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat);
uint64_t MatMulNumM64_avx512_vpopcntdq_movmask_ptr(const M64 *Mat, uint64_t n);
#endif
