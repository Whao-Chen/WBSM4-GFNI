#ifndef WBDL_GFNI_H
#define WBDL_GFNI_H

#include <immintrin.h>
#include <stdint.h>

#include "WBMatrix/WBMatrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t matrix[4][4];
    uint32_t vector;
} wbdl_gfni_aff32;

void wbdl_gfni_prepare_aff32(const Aff32 *affine,
                             wbdl_gfni_aff32 *prepared);

void wbdl_gfni_generate_table(uint32_t table[4][256],
                              const Aff8 input_encoding[4],
                              uint32_t table_key,
                              const M32 *encoded_linear,
                              const uint32_t output_xor[4]);

void wbdl_gfni_sbox_bytes(const uint8_t input[64], uint8_t output[64]);

static inline uint8_t wbdl_gfni_xor_qword_low_bytes(__m256i value)
{
    const __m256i pair_xor =
        _mm256_xor_si256(value, _mm256_srli_si256(value, 8));
    const uint8_t low = (uint8_t)_mm_cvtsi128_si32(
        _mm256_castsi256_si128(pair_xor));
    const uint8_t high = (uint8_t)_mm_cvtsi128_si32(
        _mm256_extracti128_si256(pair_xor, 1));

    return (uint8_t)(low ^ high);
}

static inline uint32_t wbdl_gfni_apply_aff32(
    const wbdl_gfni_aff32 *prepared, uint32_t input)
{
    const __m256i bytes = _mm256_set_epi64x(
        (long long)(uint8_t)input,
        (long long)(uint8_t)(input >> 8),
        (long long)(uint8_t)(input >> 16),
        (long long)(uint8_t)(input >> 24));
    const __m256i matrix0 = _mm256_loadu_si256(
        (const __m256i *)(const void *)prepared->matrix[0]);
    const __m256i matrix1 = _mm256_loadu_si256(
        (const __m256i *)(const void *)prepared->matrix[1]);
    const __m256i matrix2 = _mm256_loadu_si256(
        (const __m256i *)(const void *)prepared->matrix[2]);
    const __m256i matrix3 = _mm256_loadu_si256(
        (const __m256i *)(const void *)prepared->matrix[3]);
    const uint32_t output =
        ((uint32_t)wbdl_gfni_xor_qword_low_bytes(
             _mm256_gf2p8affine_epi64_epi8(bytes, matrix0, 0)) << 24) |
        ((uint32_t)wbdl_gfni_xor_qword_low_bytes(
             _mm256_gf2p8affine_epi64_epi8(bytes, matrix1, 0)) << 16) |
        ((uint32_t)wbdl_gfni_xor_qword_low_bytes(
             _mm256_gf2p8affine_epi64_epi8(bytes, matrix2, 0)) << 8) |
        (uint32_t)wbdl_gfni_xor_qword_low_bytes(
             _mm256_gf2p8affine_epi64_epi8(bytes, matrix3, 0));

    return output ^ prepared->vector;
}

#ifdef __cplusplus
}
#endif

#endif
