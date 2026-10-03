#ifndef XL_WBSM4_SPECIALIZED_H
#define XL_WBSM4_SPECIALIZED_H

#include <immintrin.h>
#include <stdint.h>

#include <WBMatrix/WBMatrix.h>

typedef struct {
    __m256i matrices;
    uint32_t vector;
} XLWBSM4Affine8x4Prepared;

typedef struct {
    __m256i transposed_blocks[4];
} XLWBSM4M32PreparedRight;

typedef struct {
    __m256i byte_maps[4];
} XLWBSM4QBoxLinearPrepared;

void xl_wbsm4_affine8x4_prepare(
    const Aff8 affine[4], XLWBSM4Affine8x4Prepared *prepared);
uint32_t xl_wbsm4_affine8x4_apply(
    const XLWBSM4Affine8x4Prepared *prepared, uint8_t input);

void xl_wbsm4_m32_prepare_right(
    const M32 *right, XLWBSM4M32PreparedRight *prepared);
void xl_wbsm4_m32_multiply_prepared(
    const M32 *left, const XLWBSM4M32PreparedRight *prepared, M32 *product);

void xl_wbsm4_affinemix_same_left3(
    const Aff32 *left,
    const Aff32 *right0, const Aff32 *right1, const Aff32 *right2,
    Aff32 *output0, Aff32 *output1, Aff32 *output2);
void xl_wbsm4_affinemix_same_left2_without_second_left_vector(
    const Aff32 *left, const Aff32 *right0, const Aff32 *right1,
    Aff32 *output0, Aff32 *output1);

void xl_wbsm4_qbox_prepare(
    const M32 *q_matrix, const XLWBSM4M32PreparedRight *prepared_csl,
    XLWBSM4QBoxLinearPrepared *prepared);
void xl_wbsm4_qbox_apply4(
    const XLWBSM4QBoxLinearPrepared *prepared,
    const uint8_t inputs[4], uint32_t outputs[4]);

#endif
