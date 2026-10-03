#include "baiwu_gfni_simd.h"

#include "wbsm4.h"

#include <immintrin.h>

typedef struct {
    uint64_t matrix[4][8];
    uint8_t vector[4];
} prepared_aff32;

static prepared_aff32 prepared_se[4];
static prepared_aff32 prepared_fe[4];

static void prepare_aff32(const Aff32 *affine, prepared_aff32 *prepared)
{
    int output_byte;
    for (output_byte = 0; output_byte < 4; ++output_byte) {
        int input_byte;
        for (input_byte = 0; input_byte < 4; ++input_byte) {
            uint64_t rows = 0;
            int row;
            for (row = 0; row < 8; ++row) {
                rows |= (uint64_t)(uint8_t)(
                    affine->Mat.M[output_byte * 8 + row] >>
                    (24 - 8 * input_byte)) << (8 * row);
            }
            prepared->matrix[output_byte][2 * input_byte] = rows;
            prepared->matrix[output_byte][2 * input_byte + 1] = rows;
        }
        prepared->vector[output_byte] = (uint8_t)(
            affine->Vec.V >> (24 - 8 * output_byte));
    }
}

void baiwu_gfni_prepare(void)
{
    int word;
    for (word = 0; word < 4; ++word) {
        prepare_aff32(&SE[word], &prepared_se[word]);
        prepare_aff32(&FE[word], &prepared_fe[word]);
    }
}

__attribute__((noinline))
static __m512i apply_aff32_16(const prepared_aff32 *affine, __m512i input)
{
    __m128i input_byte[4];
    __m512i packed_input;
    __m512i output = _mm512_setzero_si512();
    int output_byte;

    input_byte[0] = _mm512_cvtepi32_epi8(_mm512_srli_epi32(input, 24));
    input_byte[1] = _mm512_cvtepi32_epi8(_mm512_srli_epi32(input, 16));
    input_byte[2] = _mm512_cvtepi32_epi8(_mm512_srli_epi32(input, 8));
    input_byte[3] = _mm512_cvtepi32_epi8(input);

    packed_input = _mm512_castsi128_si512(input_byte[0]);
    packed_input = _mm512_inserti32x4(
        packed_input, input_byte[1], 1);
    packed_input = _mm512_inserti32x4(
        packed_input, input_byte[2], 2);
    packed_input = _mm512_inserti32x4(
        packed_input, input_byte[3], 3);

    for (output_byte = 0; output_byte < 4; ++output_byte) {
        const __m512i transformed = _mm512_gf2p8affine_epi64_epi8(
            packed_input,
            _mm512_loadu_si512(
                (const void *)affine->matrix[output_byte]),
            0);
        __m128i byte_result = _mm512_castsi512_si128(transformed);

        byte_result = _mm_xor_si128(
            byte_result, _mm512_extracti32x4_epi32(transformed, 1));
        byte_result = _mm_xor_si128(
            byte_result, _mm512_extracti32x4_epi32(transformed, 2));
        byte_result = _mm_xor_si128(
            byte_result, _mm512_extracti32x4_epi32(transformed, 3));
        byte_result = _mm_xor_si128(
            byte_result, _mm_set1_epi8((char)affine->vector[output_byte]));
        {
            __m512i widened = _mm512_cvtepu8_epi32(byte_result);
            if (output_byte != 3) {
                widened = _mm512_slli_epi32(
                    widened, 24 - 8 * output_byte);
            }
            output = _mm512_or_si512(output, widened);
        }
    }
    return output;
}

static inline __m512i byte_swap32(__m512i value)
{
    const __m128i mask128 = _mm_setr_epi8(
         3,  2,  1,  0,  7,  6,  5,  4,
        11, 10,  9,  8, 15, 14, 13, 12);
    return _mm512_shuffle_epi8(value, _mm512_broadcast_i32x4(mask128));
}

static inline __m512i word_indices(int word)
{
    const __m512i base = _mm512_setr_epi32(
         0,  4,  8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60);
    return _mm512_add_epi32(base, _mm512_set1_epi32(word));
}

static inline __m512i load_word16(const uint8_t *input, int word)
{
    return byte_swap32(_mm512_i32gather_epi32(
        word_indices(word), (const void *)input, 4));
}

static inline void store_word16(uint8_t *output, int word, __m512i value)
{
    _mm512_i32scatter_epi32(
        (void *)output, word_indices(word), byte_swap32(value), 4);
}

static inline __m512i extract_byte32(__m512i value, int byte_index)
{
    if (byte_index != 3) {
        value = _mm512_srli_epi32(value, 24 - 8 * byte_index);
    }
    return _mm512_and_si512(value, _mm512_set1_epi32(0xff));
}

static inline __m512i gather_td(int round, int branch, int byte_index,
                                __m512i value)
{
    return _mm512_i32gather_epi32(
        extract_byte32(value, byte_index),
        (const void *)TD[round][branch][byte_index], 4);
}

void baiwu_gfni_encrypt16(const uint8_t input[16 * 16],
                          uint8_t output[16 * 16])
{
    __m512i x0 = apply_aff32_16(&prepared_se[0], load_word16(input, 0));
    __m512i x1 = apply_aff32_16(&prepared_se[1], load_word16(input, 1));
    __m512i x2 = apply_aff32_16(&prepared_se[2], load_word16(input, 2));
    __m512i x3 = apply_aff32_16(&prepared_se[3], load_word16(input, 3));
    int round;

    for (round = 0; round < 32; ++round) {
        __m512i state[3];
        __m512i s0 = _mm512_setzero_si512();
        __m512i s1 = _mm512_setzero_si512();
        __m512i x4 = _mm512_setzero_si512();
        int branch;
        int byte_index;

        state[0] = x1;
        state[1] = x2;
        state[2] = x3;
        for (branch = 1; branch < 4; ++branch) {
            s0 = _mm512_xor_si512(
                s0, gather_td(round, branch, 0, state[branch - 1]));
            s0 = _mm512_xor_si512(
                s0, gather_td(round, branch, 1, state[branch - 1]));
            s1 = _mm512_xor_si512(
                s1, gather_td(round, branch, 2, state[branch - 1]));
            s1 = _mm512_xor_si512(
                s1, gather_td(round, branch, 3, state[branch - 1]));
        }
        for (byte_index = 0; byte_index < 4; ++byte_index) {
            const __m512i tr_index = _mm512_or_si512(
                _mm512_slli_epi32(extract_byte32(s0, byte_index), 8),
                extract_byte32(s1, byte_index));
            x4 = _mm512_xor_si512(
                x4,
                _mm512_i32gather_epi32(
                    tr_index, (const void *)TR[round][byte_index][0], 4));
            x4 = _mm512_xor_si512(
                x4, gather_td(round, 0, byte_index, x0));
        }

        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }

    store_word16(output, 0, apply_aff32_16(&prepared_fe[0], x3));
    store_word16(output, 1, apply_aff32_16(&prepared_fe[1], x2));
    store_word16(output, 2, apply_aff32_16(&prepared_fe[2], x1));
    store_word16(output, 3, apply_aff32_16(&prepared_fe[3], x0));
}

__attribute__((optimize("no-tree-vectorize")))
void baiwu_interleaved_encrypt16(const uint8_t input[16 * 16],
                                 uint8_t output[16 * 16])
{
    uint32_t x0[16];
    uint32_t x1[16];
    uint32_t x2[16];
    uint32_t x3[16];
    int round;

    _mm512_storeu_si512(
        (void *)x0,
        apply_aff32_16(&prepared_se[0], load_word16(input, 0)));
    _mm512_storeu_si512(
        (void *)x1,
        apply_aff32_16(&prepared_se[1], load_word16(input, 1)));
    _mm512_storeu_si512(
        (void *)x2,
        apply_aff32_16(&prepared_se[2], load_word16(input, 2)));
    _mm512_storeu_si512(
        (void *)x3,
        apply_aff32_16(&prepared_se[3], load_word16(input, 3)));

    for (round = 0; round < 32; ++round) {
        int lane;
        for (lane = 0; lane < 16; ++lane) {
            const uint32_t state[3] = {x1[lane], x2[lane], x3[lane]};
            uint32_t s0 = 0;
            uint32_t s1 = 0;
            uint32_t x4 = 0;
            int branch;
            int byte_index;

            for (branch = 1; branch < 4; ++branch) {
                const uint32_t value = state[branch - 1];
                s0 ^= TD[round][branch][0][value >> 24];
                s0 ^= TD[round][branch][1][(value >> 16) & 0xff];
                s1 ^= TD[round][branch][2][(value >> 8) & 0xff];
                s1 ^= TD[round][branch][3][value & 0xff];
            }
            for (byte_index = 0; byte_index < 4; ++byte_index) {
                const int shift = 24 - 8 * byte_index;
                const uint8_t first = (uint8_t)(s0 >> shift);
                const uint8_t second = (uint8_t)(s1 >> shift);
                const uint8_t old_state = (uint8_t)(x0[lane] >> shift);
                x4 ^= TR[round][byte_index][first][second];
                x4 ^= TD[round][0][byte_index][old_state];
            }

            x0[lane] = x1[lane];
            x1[lane] = x2[lane];
            x2[lane] = x3[lane];
            x3[lane] = x4;
        }
    }

    store_word16(
        output, 0,
        apply_aff32_16(
            &prepared_fe[0], _mm512_loadu_si512((const void *)x3)));
    store_word16(
        output, 1,
        apply_aff32_16(
            &prepared_fe[1], _mm512_loadu_si512((const void *)x2)));
    store_word16(
        output, 2,
        apply_aff32_16(
            &prepared_fe[2], _mm512_loadu_si512((const void *)x1)));
    store_word16(
        output, 3,
        apply_aff32_16(
            &prepared_fe[3], _mm512_loadu_si512((const void *)x0)));
}
