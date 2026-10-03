#ifndef WBDL_MATRIX_BINDINGS_H
#define WBDL_MATRIX_BINDINGS_H
#include "WBMatrix/WBMatrix.h"
#ifdef WBSM4_GFNI_BACKEND
#include "matrix_backend.h"
#define affineU8(a,x) wbdl_opt_affine_u8(&(a),(x))
#define affineU32(a,x) wbdl_opt_affine_u32(&(a),(x))
#define MatMulNumM32(a,x) wbdl_opt_mat_mul_num32(&(a),(x))
#define MatMulMatM32(a,b,r) wbdl_opt_mat_mul_mat32(&(a),&(b),(r))
#define genaffinepairM8(a,b) wbdl_opt_gen_affine8((a),(b))
#define genaffinepairM32(a,b) wbdl_opt_gen_affine32((a),(b))
#define affinemixM32(a,b,r) wbdl_opt_affine_mix32(&(a),&(b),(r))
#endif
#endif
