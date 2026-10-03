#include "wbdl_gfni.h"

#include <string.h>

static const uint8_t WBDL_GFNI_SEQUENCE[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};

static const uint64_t SM4_SBOX_PRE_MATRIX = UINT64_C(0x4c287db91a22505d);
static const uint64_t SM4_SBOX_POST_MATRIX = UINT64_C(0xf3ab34a974a6b589);

static inline __m512i sm4_sbox_zmm(__m512i input)
{
    const __m512i pre_matrix = _mm512_set1_epi64(
        (long long)SM4_SBOX_PRE_MATRIX);
    const __m512i post_matrix = _mm512_set1_epi64(
        (long long)SM4_SBOX_POST_MATRIX);
    const __m512i pre = _mm512_gf2p8affine_epi64_epi8(
        input, pre_matrix, 0x3e);

    return _mm512_gf2p8affineinv_epi64_epi8(
        pre, post_matrix, 0xd3);
}

static inline __m256i pack_m8_block_matrices(const M32 *matrix,
                                              int output_byte)
{
    uint64_t blocks[4];
    int input_byte;
    int row;

    for (input_byte = 0; input_byte < 4; ++input_byte) {
        uint64_t rows = 0;

        for (row = 0; row < 8; ++row) {
            const uint8_t matrix_row = (uint8_t)(
                matrix->M[output_byte * 8 + row] >>
                (24 - 8 * input_byte));
            rows |= (uint64_t)matrix_row << (8 * row);
        }
        blocks[input_byte] = rows;
    }

    return _mm256_loadu_si256((const __m256i *)(const void *)blocks);
}

void wbdl_gfni_prepare_aff32(const Aff32 *affine,
                             wbdl_gfni_aff32 *prepared)
{
    int output_byte;

    for (output_byte = 0; output_byte < 4; ++output_byte) {
        const __m256i packed = pack_m8_block_matrices(
            &affine->Mat, output_byte);
        _mm256_storeu_si256(
            (__m256i *)(void *)prepared->matrix[output_byte], packed);
    }
    prepared->vector = affine->Vec.V;
}

static inline uint64_t extract_8_to_8_matrix(const M32 *matrix,
                                              int input_byte,
                                              int output_byte)
{
    uint64_t rows = 0;
    int row;

    for (row = 0; row < 8; ++row) {
        const uint8_t matrix_row = (uint8_t)(
            matrix->M[output_byte * 8 + row] >>
            (24 - 8 * input_byte));
        rows |= (uint64_t)matrix_row << (8 * row);
    }
    return rows;
}

static inline void store_table_words(uint32_t output[256], int offset,
                                     __m512i byte0, __m512i byte1,
                                     __m512i byte2, __m512i byte3,
                                     uint32_t xor_mask)
{
#define WBDL_GFNI_STORE_16(CHUNK) do {                                      \
        const __m128i input0 = _mm512_extracti32x4_epi32(byte0, (CHUNK));   \
        const __m128i input1 = _mm512_extracti32x4_epi32(byte1, (CHUNK));   \
        const __m128i input2 = _mm512_extracti32x4_epi32(byte2, (CHUNK));   \
        const __m128i input3 = _mm512_extracti32x4_epi32(byte3, (CHUNK));   \
        __m512i words = _mm512_slli_epi32(                                  \
            _mm512_cvtepu8_epi32(input0), 24);                              \
        words = _mm512_or_si512(words, _mm512_slli_epi32(                   \
            _mm512_cvtepu8_epi32(input1), 16));                             \
        words = _mm512_or_si512(words, _mm512_slli_epi32(                   \
            _mm512_cvtepu8_epi32(input2), 8));                              \
        words = _mm512_or_si512(words, _mm512_cvtepu8_epi32(input3));       \
        words = _mm512_xor_si512(words, _mm512_set1_epi32((int)xor_mask));  \
        _mm512_storeu_si512((void *)(output + offset + 16 * (CHUNK)),       \
                            words);                                         \
    } while (0)

    WBDL_GFNI_STORE_16(0);
    WBDL_GFNI_STORE_16(1);
    WBDL_GFNI_STORE_16(2);
    WBDL_GFNI_STORE_16(3);

#undef WBDL_GFNI_STORE_16
}

void wbdl_gfni_generate_table(uint32_t table[4][256],
                              const Aff8 input_encoding[4],
                              uint32_t table_key,
                              const M32 *encoded_linear,
                              const uint32_t output_xor[4])
{
    const __m512i sequence = _mm512_loadu_si512(
        (const void *)WBDL_GFNI_SEQUENCE);
    int input_byte;

    for (input_byte = 0; input_byte < 4; ++input_byte) {
        uint64_t input_rows;
        __m512i input_matrix;
        __m512i key;
        __m512i output_matrix[4];
        int output_byte;
        int batch;

        memcpy(&input_rows, input_encoding[input_byte].Mat.M,
               sizeof(input_rows));
        input_matrix = _mm512_set1_epi64((long long)input_rows);
        key = _mm512_set1_epi8((char)(uint8_t)(
            table_key >> (24 - 8 * input_byte)));
        for (output_byte = 0; output_byte < 4; ++output_byte) {
            const uint64_t rows = extract_8_to_8_matrix(
                encoded_linear, input_byte, output_byte);
            output_matrix[output_byte] = _mm512_set1_epi64((long long)rows);
        }

        for (batch = 0; batch < 4; ++batch) {
            const __m512i values = _mm512_add_epi8(
                sequence, _mm512_set1_epi8((char)(batch * 64)));
            const __m512i decoded_linear = _mm512_gf2p8affine_epi64_epi8(
                values, input_matrix, 0);
            const __m512i decoded = _mm512_xor_si512(
                decoded_linear,
                _mm512_set1_epi8((char)input_encoding[input_byte].Vec.V));
            const __m512i substituted = sm4_sbox_zmm(
                _mm512_xor_si512(decoded, key));
            const __m512i output0 = _mm512_gf2p8affine_epi64_epi8(
                substituted, output_matrix[0], 0);
            const __m512i output1 = _mm512_gf2p8affine_epi64_epi8(
                substituted, output_matrix[1], 0);
            const __m512i output2 = _mm512_gf2p8affine_epi64_epi8(
                substituted, output_matrix[2], 0);
            const __m512i output3 = _mm512_gf2p8affine_epi64_epi8(
                substituted, output_matrix[3], 0);

            store_table_words(table[input_byte], batch * 64,
                              output0, output1, output2, output3,
                              output_xor[input_byte]);
        }
    }
}

void wbdl_gfni_sbox_bytes(const uint8_t input[64], uint8_t output[64])
{
    const __m512i values = _mm512_loadu_si512((const void *)input);
    _mm512_storeu_si512((void *)output, sm4_sbox_zmm(values));
}
