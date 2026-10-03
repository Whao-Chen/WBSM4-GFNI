#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>
#include <x86intrin.h>


uint32_t affineU32_matmulnum_gpt7_wrapper_ptr(const Aff32 *aff, uint32_t arr)
{
    return MatMulNumM32_avx512_vpopcntdq_bitreverse32_ptr(&aff->Mat, arr) ^ aff->Vec.V;
}

