#include "wsise_gfni_simd.h"

#include "wbsm4.h"

#include <immintrin.h>
#include <string.h>

typedef struct {
    uint64_t matrix[4][4];
    uint8_t vector[4];
} prepared_aff32;

static prepared_aff32 prepared_m[32][3];
static prepared_aff32 prepared_c[32];
static prepared_aff32 prepared_d[32];
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
                const uint8_t matrix_row = (uint8_t)(
                    affine->Mat.M[output_byte * 8 + row] >>
                    (24 - 8 * input_byte));
                rows |= (uint64_t)matrix_row << (8 * row);
            }
            prepared->matrix[output_byte][input_byte] = rows;
        }
        prepared->vector[output_byte] = (uint8_t)(
            affine->Vec.V >> (24 - 8 * output_byte));
    }
}

void wsise_gfni_prepare(void)
{
    int round;

    for (round = 0; round < 32; ++round) {
        int input_word;

        for (input_word = 0; input_word < 3; ++input_word) {
            prepare_aff32(&M[round][input_word],
                          &prepared_m[round][input_word]);
        }
        prepare_aff32(&C[round], &prepared_c[round]);
        prepare_aff32(&D[round], &prepared_d[round]);
    }
    for (round = 0; round < 4; ++round) {
        prepare_aff32(&SE[round], &prepared_se[round]);
        prepare_aff32(&FE[round], &prepared_fe[round]);
    }
}

static inline void split_words16(__m512i input, __m128i bytes[4])
{
    bytes[0] = _mm512_cvtepi32_epi8(_mm512_srli_epi32(input, 24));
    bytes[1] = _mm512_cvtepi32_epi8(_mm512_srli_epi32(input, 16));
    bytes[2] = _mm512_cvtepi32_epi8(_mm512_srli_epi32(input, 8));
    bytes[3] = _mm512_cvtepi32_epi8(input);
}

static inline __m128i apply_linear_byte16(
    const prepared_aff32 *affine, const __m128i input[4], int output_byte)
{
    __m128i result = _mm_setzero_si128();
    int input_byte;

    for (input_byte = 0; input_byte < 4; ++input_byte) {
        result = _mm_xor_si128(
            result,
            _mm_gf2p8affine_epi64_epi8(
                input[input_byte],
                _mm_set1_epi64x((long long)
                    affine->matrix[output_byte][input_byte]),
                0));
    }
    return result;
}

static inline __m512i join_bytes16(const __m128i bytes[4])
{
    __m512i output = _mm512_setzero_si512();
    int output_byte;

    for (output_byte = 0; output_byte < 4; ++output_byte) {
        __m512i widened = _mm512_cvtepu8_epi32(bytes[output_byte]);
        if (output_byte != 3) {
            widened = _mm512_slli_epi32(
                widened, 24 - 8 * output_byte);
        }
        output = _mm512_or_si512(output, widened);
    }
    return output;
}

static inline __m512i apply_aff32_16(const prepared_aff32 *affine,
                                     __m512i input)
{
    __m128i input_bytes[4];
    __m128i output_bytes[4];
    int output_byte;

    split_words16(input, input_bytes);
    for (output_byte = 0; output_byte < 4; ++output_byte) {
        output_bytes[output_byte] = _mm_xor_si128(
            apply_linear_byte16(affine, input_bytes, output_byte),
            _mm_set1_epi8((char)affine->vector[output_byte]));
    }
    return join_bytes16(output_bytes);
}

static inline __m512i apply_aff32_xor2_16(
    const prepared_aff32 *left, __m512i left_input,
    const prepared_aff32 *right, __m512i right_input)
{
    __m128i left_bytes[4];
    __m128i right_bytes[4];
    __m128i output_bytes[4];
    int output_byte;

    split_words16(left_input, left_bytes);
    split_words16(right_input, right_bytes);
    for (output_byte = 0; output_byte < 4; ++output_byte) {
        output_bytes[output_byte] = _mm_xor_si128(
            apply_linear_byte16(left, left_bytes, output_byte),
            apply_linear_byte16(right, right_bytes, output_byte));
        output_bytes[output_byte] = _mm_xor_si128(
            output_bytes[output_byte],
            _mm_set1_epi8((char)(
                left->vector[output_byte] ^ right->vector[output_byte])));
    }
    return join_bytes16(output_bytes);
}

static inline __m512i apply_aff32_xor3_16(
    const prepared_aff32 *first, __m512i first_input,
    const prepared_aff32 *second, __m512i second_input,
    const prepared_aff32 *third, __m512i third_input)
{
    __m128i first_bytes[4];
    __m128i second_bytes[4];
    __m128i third_bytes[4];
    __m128i output_bytes[4];
    int output_byte;

    split_words16(first_input, first_bytes);
    split_words16(second_input, second_bytes);
    split_words16(third_input, third_bytes);
    for (output_byte = 0; output_byte < 4; ++output_byte) {
        output_bytes[output_byte] = _mm_xor_si128(
            apply_linear_byte16(first, first_bytes, output_byte),
            apply_linear_byte16(second, second_bytes, output_byte));
        output_bytes[output_byte] = _mm_xor_si128(
            output_bytes[output_byte],
            apply_linear_byte16(third, third_bytes, output_byte));
        output_bytes[output_byte] = _mm_xor_si128(
            output_bytes[output_byte],
            _mm_set1_epi8((char)(
                first->vector[output_byte] ^
                second->vector[output_byte] ^
                third->vector[output_byte])));
    }
    return join_bytes16(output_bytes);
}

static inline __m512i load_word16(const uint8_t *input, int word);
static inline void store_word16(uint8_t *output, int word, __m512i value);
static inline __m512i table_lookup16(int round, __m512i value);

static int check_aff32(const Aff32 *affine, const prepared_aff32 *prepared)
{
    uint32_t input[16];
    uint32_t output[16];
    int lane;

    for (lane = 0; lane < 16; ++lane) {
        input[lane] = UINT32_C(0x9e3779b9) * (uint32_t)(lane + 1);
    }
    _mm512_storeu_si512(
        (void *)output,
        apply_aff32_16(prepared, _mm512_loadu_si512((const void *)input)));
    for (lane = 0; lane < 16; ++lane) {
        if (output[lane] != affineU32(*affine, input[lane])) return 0;
    }
    return 1;
}

int wsise_gfni_selftest(void)
{
    int round;

    for (round = 0; round < 32; ++round) {
        int word;
        for (word = 0; word < 3; ++word) {
            if (!check_aff32(&M[round][word], &prepared_m[round][word])) {
                return -1;
            }
        }
        if (!check_aff32(&C[round], &prepared_c[round]) ||
            !check_aff32(&D[round], &prepared_d[round])) {
            return -1;
        }
    }
    for (round = 0; round < 4; ++round) {
        if (!check_aff32(&SE[round], &prepared_se[round]) ||
            !check_aff32(&FE[round], &prepared_fe[round])) {
            return -1;
        }
    }

    {
        uint32_t input_words[16];
        uint32_t output_words[16];
        int lane;

        for (lane = 0; lane < 16; ++lane) {
            input_words[lane] = UINT32_C(0x01030507) * (uint32_t)(lane + 1);
        }
        for (round = 0; round < 32; ++round) {
            _mm512_storeu_si512(
                (void *)output_words,
                table_lookup16(
                    round, _mm512_loadu_si512((const void *)input_words)));
            for (lane = 0; lane < 16; ++lane) {
                const uint32_t value = input_words[lane];
                const uint64_t combined =
                    Table[round][0][value >> 24] ^
                    Table[round][1][(value >> 16) & 0xff] ^
                    Table[round][2][(value >> 8) & 0xff] ^
                    Table[round][3][value & 0xff];
                if (output_words[lane] != (uint32_t)(combined >> 32)) {
                    return -2;
                }
            }
        }
    }

    {
        uint8_t input[16 * 16];
        uint8_t output[16 * 16] = {0};
        int byte;

        for (byte = 0; byte < 16 * 16; ++byte) {
            input[byte] = (uint8_t)byte;
        }
        for (round = 0; round < 4; ++round) {
            store_word16(output, round, load_word16(input, round));
        }
        if (memcmp(input, output, sizeof(input)) != 0) return -3;
    }
    return 1;
}

static inline __m512i load_word16(const uint8_t *input, int word)
{
    const __m512i block_word_indices = _mm512_setr_epi32(
         0,  4,  8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60);
    const __m512i indices = _mm512_add_epi32(
        block_word_indices, _mm512_set1_epi32(word));
    const __m512i native_words = _mm512_i32gather_epi32(
        indices, (const void *)input, 4);

    const __m128i byte_swap128 = _mm_setr_epi8(
         3,  2,  1,  0,  7,  6,  5,  4,
        11, 10,  9,  8, 15, 14, 13, 12);

    return _mm512_shuffle_epi8(
        native_words, _mm512_broadcast_i32x4(byte_swap128));
}

static inline void store_word16(uint8_t *output, int word, __m512i value)
{
    const __m512i block_word_indices = _mm512_setr_epi32(
         0,  4,  8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60);
    const __m512i indices = _mm512_add_epi32(
        block_word_indices, _mm512_set1_epi32(word));
    const __m128i byte_swap128 = _mm_setr_epi8(
         3,  2,  1,  0,  7,  6,  5,  4,
        11, 10,  9,  8, 15, 14, 13, 12);
    const __m512i native_words = _mm512_shuffle_epi8(
        value, _mm512_broadcast_i32x4(byte_swap128));

    _mm512_i32scatter_epi32((void *)output, indices, native_words, 4);
}

static inline __m512i table_lookup16(int round, __m512i value)
{
    __m512i low_result = _mm512_setzero_si512();
    __m512i high_result = _mm512_setzero_si512();
    __m512i indices[4];
    int table_index;

    indices[0] = _mm512_srli_epi32(value, 24);
    indices[1] = _mm512_and_si512(
        _mm512_srli_epi32(value, 16), _mm512_set1_epi32(0xff));
    indices[2] = _mm512_and_si512(
        _mm512_srli_epi32(value, 8), _mm512_set1_epi32(0xff));
    indices[3] = _mm512_and_si512(value, _mm512_set1_epi32(0xff));

    for (table_index = 0; table_index < 4; ++table_index) {
        const __m256i low_indices = _mm512_castsi512_si256(
            indices[table_index]);
        const __m256i high_indices = _mm512_extracti64x4_epi64(
            indices[table_index], 1);

        low_result = _mm512_xor_si512(
            low_result,
            _mm512_i32gather_epi64(
                low_indices, (const void *)Table[round][table_index], 8));
        high_result = _mm512_xor_si512(
            high_result,
            _mm512_i32gather_epi64(
                high_indices, (const void *)Table[round][table_index], 8));
    }

    {
        const __m256i low_words = _mm512_cvtepi64_epi32(
            _mm512_srli_epi64(low_result, 32));
        const __m256i high_words = _mm512_cvtepi64_epi32(
            _mm512_srli_epi64(high_result, 32));
        __m512i result = _mm512_castsi256_si512(low_words);
        return _mm512_inserti64x4(result, high_words, 1);
    }
}

void wsise_gfni_encrypt16(const uint8_t input[16 * 16],
                          uint8_t output[16 * 16])
{
    __m512i x0 = apply_aff32_16(&prepared_se[0], load_word16(input, 0));
    __m512i x1 = apply_aff32_16(&prepared_se[1], load_word16(input, 1));
    __m512i x2 = apply_aff32_16(&prepared_se[2], load_word16(input, 2));
    __m512i x3 = apply_aff32_16(&prepared_se[3], load_word16(input, 3));
    int round;

    for (round = 0; round < 32; ++round) {
        __m512i mixed = apply_aff32_xor3_16(
            &prepared_m[round][0], x1,
            &prepared_m[round][1], x2,
            &prepared_m[round][2], x3);
        __m512i x4;

        x4 = table_lookup16(round, mixed);
        x4 = apply_aff32_xor2_16(
            &prepared_c[round], x0, &prepared_d[round], x4);

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
