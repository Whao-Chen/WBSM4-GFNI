#include "masked_wbsm4_enhanced.h"

#include "WBMatrix/WBMatrix.h"
#include "sbox.h"
#include "sm4.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define ENHANCED_ROUNDS 32
#define TABLE2_ROUNDS 31
#define TABLE2_ENTRIES 65536u
#define TABLE2_BYTES \
    ((size_t)TABLE2_ROUNDS * 4u * (size_t)TABLE2_ENTRIES)
#define MASKED_WBSM4_ENHANCED_MAGIC UINT32_C(0x4d574245)

#ifndef MASKED_GFNI_TABLE2
#define MASKED_GFNI_TABLE2 1
#endif
#ifndef MASKED_GFNI_VERIFY_TABLE2
#define MASKED_GFNI_VERIFY_TABLE2 0
#endif
#ifndef MASKED_PRECOMPOSE_MASK_BRANCH
#define MASKED_PRECOMPOSE_MASK_BRANCH 0
#endif
#ifndef MASKED_PRECOMPOSE_MASK_LOOKUP_U8
#define MASKED_PRECOMPOSE_MASK_LOOKUP_U8 0
#endif
#ifndef MASKED_GFNI_VERIFY_MASK_BRANCH
#define MASKED_GFNI_VERIFY_MASK_BRANCH 0
#endif
#ifndef MASKED_PIPELINE_32WAY
#define MASKED_PIPELINE_32WAY 0
#endif
#ifndef MASKED_FUSE_U8_MASK_AFFINE
#define MASKED_FUSE_U8_MASK_AFFINE 0
#endif

typedef struct {
    uint64_t matrix[4][4];
    uint8_t vector[4];
} prepared_aff32;

struct masked_wbsm4_enhanced_context {
    Aff32 part1[ENHANCED_ROUNDS][3];
    Aff32 round_x[ENHANCED_ROUNDS];
    Aff32 masked_data[ENHANCED_ROUNDS];
    Aff32 mask_branch[ENHANCED_ROUNDS];
    Aff32 input_encoding[4];
    Aff32 output_decoding[4];
    M32 mask_decode;

    uint8_t table1[4][256];
    uint8_t *table2;
    uint8_t mask_table[ENHANCED_ROUNDS][256];
    uint8_t phi[ENHANCED_ROUNDS][256];
#if MASKED_PRECOMPOSE_MASK_BRANCH
    uint32_t simd_mask_branch[ENHANCED_ROUNDS][4][256];
#endif
#if MASKED_PRECOMPOSE_MASK_LOOKUP_U8
    uint8_t simd_mask_lookup_u8[ENHANCED_ROUNDS][4][256];
#endif

    prepared_aff32 prepared_part1[ENHANCED_ROUNDS][3];
    prepared_aff32 prepared_round_x[ENHANCED_ROUNDS];
    prepared_aff32 prepared_masked_data[ENHANCED_ROUNDS];
    prepared_aff32 prepared_mask_branch[ENHANCED_ROUNDS];
    prepared_aff32 prepared_input_encoding[4];
    prepared_aff32 prepared_output_decoding[4];
    prepared_aff32 prepared_mask_decode;

    uint32_t magic;
};

static uint32_t load_be32(const uint8_t input[4])
{
    return ((uint32_t)input[0] << 24) |
           ((uint32_t)input[1] << 16) |
           ((uint32_t)input[2] << 8) |
           (uint32_t)input[3];
}

static void store_be32(uint32_t value, uint8_t output[4])
{
    output[0] = (uint8_t)(value >> 24);
    output[1] = (uint8_t)(value >> 16);
    output[2] = (uint8_t)(value >> 8);
    output[3] = (uint8_t)value;
}

static const uint8_t GFNI_SEQUENCE[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};

__attribute__((noinline))
static __m512i sm4_sbox_zmm(__m512i input)
{
    const __m512i pre_matrix = _mm512_set1_epi64(
        (long long)UINT64_C(0x4c287db91a22505d));
    const __m512i post_matrix = _mm512_set1_epi64(
        (long long)UINT64_C(0xf3ab34a974a6b589));
    const __m512i pre = _mm512_gf2p8affine_epi64_epi8(
        input, pre_matrix, 0x3e);

    return _mm512_gf2p8affineinv_epi64_epi8(
        pre, post_matrix, 0xd3);
}

__attribute__((noinline))
static __m512i matmul_m8_zmm(const M8 *matrix, __m512i input)
{
    uint64_t rows;
    memcpy(&rows, matrix->M, sizeof(rows));
    return _mm512_gf2p8affine_epi64_epi8(
        input, _mm512_set1_epi64((long long)rows), 0);
}

__attribute__((noinline))
static __m512i affine_u8_zmm(const Aff8 *affine, __m512i input)
{
    uint64_t rows;
    memcpy(&rows, affine->Mat.M, sizeof(rows));
    return _mm512_xor_si512(
        _mm512_gf2p8affine_epi64_epi8(
            input, _mm512_set1_epi64((long long)rows), 0),
        _mm512_set1_epi8((char)affine->Vec.V));
}

static int validate_gfni_primitives(const M8 *matrix, const Aff8 *affine)
{
    uint8_t input[64];
    uint8_t output[64];
    int value;

    for (value = 0; value < 64; ++value) input[value] = (uint8_t)value;
    _mm512_storeu_si512(
        (void *)output,
        sm4_sbox_zmm(_mm512_loadu_si512((const void *)input)));
    for (value = 0; value < 64; ++value) {
        if (output[value] != SBOX[input[value]]) {
            return MASKED_WBSM4_ENHANCED_GFNI_MISMATCH;
        }
    }

    _mm512_storeu_si512(
        (void *)output,
        matmul_m8_zmm(matrix, _mm512_loadu_si512((const void *)input)));
    for (value = 0; value < 64; ++value) {
        if (output[value] != MatMulNumM8(*matrix, input[value])) {
            return MASKED_WBSM4_ENHANCED_GFNI_MISMATCH;
        }
    }

    _mm512_storeu_si512(
        (void *)output,
        affine_u8_zmm(affine, _mm512_loadu_si512((const void *)input)));
    for (value = 0; value < 64; ++value) {
        if (output[value] != affineU8(*affine, input[value])) {
            return MASKED_WBSM4_ENHANCED_GFNI_MISMATCH;
        }
    }
    return MASKED_WBSM4_ENHANCED_OK;
}

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

static void prepare_matrix32(const M32 *matrix, prepared_aff32 *prepared)
{
    Aff32 affine;
    affine.Mat = *matrix;
    affine.Vec.V = 0;
    prepare_aff32(&affine, prepared);
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

__attribute__((noinline))
static __m512i apply_aff32_xor2_16(
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

__attribute__((noinline))
static __m512i apply_aff32_xor3_16(
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

static inline __m512i byte_swap32(__m512i value)
{
    const __m128i mask128 = _mm_setr_epi8(
         3,  2,  1,  0,  7,  6,  5,  4,
        11, 10,  9,  8, 15, 14, 13, 12);
    return _mm512_shuffle_epi8(value, _mm512_broadcast_i32x4(mask128));
}

static inline __m512i load_word16(const uint8_t *input, int word)
{
    const __m512i base_indices = _mm512_setr_epi32(
         0,  4,  8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60);
    const __m512i indices = _mm512_add_epi32(
        base_indices, _mm512_set1_epi32(word));
    return byte_swap32(_mm512_i32gather_epi32(
        indices, (const void *)input, 4));
}

static inline void store_word16(uint8_t *output, int word, __m512i value)
{
    const __m512i base_indices = _mm512_setr_epi32(
         0,  4,  8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60);
    const __m512i indices = _mm512_add_epi32(
        base_indices, _mm512_set1_epi32(word));
    _mm512_i32scatter_epi32(
        (void *)output, indices, byte_swap32(value), 4);
}

static inline __m512i extract_byte32(__m512i value, int byte_index)
{
    if (byte_index != 3) {
        value = _mm512_srli_epi32(value, 24 - 8 * byte_index);
    }
    return _mm512_and_si512(value, _mm512_set1_epi32(0xff));
}

static inline __m512i gather_u8_16(const uint8_t *table, __m512i indices)
{
    const __m512i words = _mm512_i32gather_epi32(
        _mm512_srli_epi32(indices, 2), (const void *)table, 4);
    const __m512i shifts = _mm512_slli_epi32(
        _mm512_and_si512(indices, _mm512_set1_epi32(3)), 3);
    return _mm512_and_si512(
        _mm512_srlv_epi32(words, shifts), _mm512_set1_epi32(0xff));
}

static inline __m512i insert_byte32(__m512i output, __m512i byte_value,
                                    int byte_index)
{
    if (byte_index != 3) {
        byte_value = _mm512_slli_epi32(
            byte_value, 24 - 8 * byte_index);
    }
    return _mm512_or_si512(output, byte_value);
}

static uint32_t rotl32(uint32_t value, unsigned int amount)
{
    return (value << amount) | (value >> (32u - amount));
}

static uint32_t sm4_linear(uint32_t value)
{
    return value ^ rotl32(value, 2) ^ rotl32(value, 10) ^
           rotl32(value, 18) ^ rotl32(value, 24);
}

static void build_sm4_linear_matrix(M32 *matrix)
{
    int input_bit;
    int output_bit;

    initM32(matrix);
    for (input_bit = 0; input_bit < 32; ++input_bit) {
        uint32_t input_basis = UINT32_C(0x80000000) >> input_bit;
        uint32_t output = sm4_linear(input_basis);

        for (output_bit = 0; output_bit < 32; ++output_bit) {
            uint32_t output_basis = UINT32_C(0x80000000) >> output_bit;
            if ((output & output_basis) != 0) {
                matrix->M[output_bit] |= input_basis;
            }
        }
    }
}

static Aff32 make_aff32(M32 matrix, uint32_t vector)
{
    Aff32 result;
    result.Mat = matrix;
    result.Vec.V = vector;
    return result;
}

static Aff32 compose_aff32(Aff32 outer, Aff32 inner)
{
    Aff32 result;
    affinemixM32(outer, inner, &result);
    return result;
}

static uint8_t phi_power(const uint8_t phi[256],
                         uint8_t value,
                         int power)
{
    int i;
    for (i = 0; i < power; ++i) {
        value = phi[value];
    }
    return value;
}

static int permutation_is_valid(const uint8_t phi[256])
{
    uint8_t seen[256] = {0};
    int phi2_identity = 1;
    int phi3_identity = 1;
    int value;

    for (value = 0; value < 256; ++value) {
        if (seen[phi[value]] != 0) {
            return 0;
        }
        seen[phi[value]] = 1;
        if (phi_power(phi, (uint8_t)value, 2) != (uint8_t)value) {
            phi2_identity = 0;
        }
        if (phi_power(phi, (uint8_t)value, 3) != (uint8_t)value) {
            phi3_identity = 0;
        }
    }
    return !phi2_identity && !phi3_identity;
}

static void generate_permutation(uint8_t phi[256])
{
    int i;

    do {
        for (i = 0; i < 256; ++i) {
            phi[i] = (uint8_t)i;
        }
        for (i = 255; i > 0; --i) {
            int other = (int)(cus_random() % (unsigned int)(i + 1));
            uint8_t temporary = phi[i];
            phi[i] = phi[other];
            phi[other] = temporary;
        }
    } while (!permutation_is_valid(phi));
}

static uint8_t recover_round_mask(
    const masked_wbsm4_enhanced_context *ctx,
    const Aff8 mask_encoding[ENHANCED_ROUNDS],
    int round,
    int byte_index,
    uint8_t table_input)
{
    uint8_t index = phi_power(ctx->phi[round], table_input, byte_index);
    uint8_t encoded = ctx->mask_table[round][index];
    return affineU8(mask_encoding[round], encoded);
}

static size_t table2_offset(int round, int byte_index, uint16_t input)
{
    return (((size_t)(round - 1) * 4u + (size_t)byte_index) *
            (size_t)TABLE2_ENTRIES) + (size_t)input;
}

static void prepare_simd_context(masked_wbsm4_enhanced_context *ctx)
{
    int round;

    for (round = 0; round < ENHANCED_ROUNDS; ++round) {
        int word;
        for (word = 0; word < 3; ++word) {
            prepare_aff32(&ctx->part1[round][word],
                          &ctx->prepared_part1[round][word]);
        }
        prepare_aff32(&ctx->round_x[round],
                      &ctx->prepared_round_x[round]);
        prepare_aff32(&ctx->masked_data[round],
                      &ctx->prepared_masked_data[round]);
        prepare_aff32(&ctx->mask_branch[round],
                      &ctx->prepared_mask_branch[round]);
    }
    for (round = 0; round < 4; ++round) {
        prepare_aff32(&ctx->input_encoding[round],
                      &ctx->prepared_input_encoding[round]);
        prepare_aff32(&ctx->output_decoding[round],
                      &ctx->prepared_output_decoding[round]);
    }
    prepare_matrix32(&ctx->mask_decode, &ctx->prepared_mask_decode);
}

masked_wbsm4_enhanced_context *masked_wbsm4_enhanced_create(void)
{
    masked_wbsm4_enhanced_context *ctx = calloc(1, sizeof(*ctx));

    if (ctx == NULL) {
        return NULL;
    }
    ctx->table2 = malloc(TABLE2_BYTES);
    if (ctx->table2 == NULL) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void masked_wbsm4_enhanced_destroy(masked_wbsm4_enhanced_context *ctx)
{
    if (ctx != NULL) {
        free(ctx->table2);
        memset(ctx, 0, sizeof(*ctx));
        free(ctx);
    }
}

int masked_wbsm4_enhanced_generate_direction(masked_wbsm4_enhanced_context *ctx,
                                   const uint8_t key[16],
                                   uint32_t random_seed, int decrypt)
{
    Aff32 p[36];
    Aff32 p_inv[36];
    Aff8 e[ENHANCED_ROUNDS][4];
    Aff8 e_inv[ENHANCED_ROUNDS][4];
    Aff32 e32_inv[ENHANCED_ROUNDS];
    Aff8 e_prime[ENHANCED_ROUNDS][4];
    Aff8 e_prime_inv[ENHANCED_ROUNDS][4];
    Aff8 mask_encoding[ENHANCED_ROUNDS];
    Aff8 mask_encoding_inv[ENHANCED_ROUNDS];
    Aff8 n[4];
    Aff8 n_inv[4];
    Aff32 n32;
    Aff32 n32_inv;
    M32 linear_matrix;
    Aff32 linear_affine;
    sm4_context key_schedule;
    uint8_t *table2;
    int i;
    int j;
    int x;

    if (ctx == NULL || key == NULL || ctx->table2 == NULL) {
        return MASKED_WBSM4_ENHANCED_BAD_ARGUMENT;
    }

    table2 = ctx->table2;
    memset(ctx, 0, sizeof(*ctx));
    ctx->table2 = table2;
    memset(ctx->table2, 0, TABLE2_BYTES);

    SetRandSeed(random_seed);
    InitRandom(random_seed);
if (decrypt) sm4_setkey_dec(&key_schedule, (unsigned char *)key);
    else sm4_setkey_enc(&key_schedule, (unsigned char *)key);
    build_sm4_linear_matrix(&linear_matrix);
    linear_affine = make_aff32(linear_matrix, 0);

    for (i = 0; i < 36; ++i) {
        genaffinepairM32(&p[i], &p_inv[i]);
    }

    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        for (j = 0; j < 4; ++j) {
            genaffinepairM8(&e[i][j], &e_inv[i][j]);
        }
        affinecomM8to32(e_inv[i][0], e_inv[i][1],
                        e_inv[i][2], e_inv[i][3], &e32_inv[i]);

        for (j = 0; j < 3; ++j) {
            ctx->part1[i][j] =
                compose_aff32(e32_inv[i], p_inv[i + j + 1]);
        }
    }

    for (j = 0; j < 4; ++j) {
        genMatpairM8(&n[j].Mat, &n_inv[j].Mat);
        n[j].Vec.V = 0;
        n_inv[j].Vec.V = 0;
    }
    affinecomM8to32(n[0], n[1], n[2], n[3], &n32);
    affinecomM8to32(n_inv[0], n_inv[1], n_inv[2], n_inv[3],
                    &n32_inv);
    ctx->mask_decode = n32.Mat;

    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        for (j = 0; j < 4; ++j) {
            genaffinepairM8(&e_prime[i][j], &e_prime_inv[i][j]);
        }
        genaffinepairM8(&mask_encoding[i], &mask_encoding_inv[i]);
        generate_permutation(ctx->phi[i]);

        for (x = 0; x < 256; ++x) {
            uint8_t mask = (uint8_t)cus_random();
            ctx->mask_table[i][x] =
                affineU8(mask_encoding_inv[i], mask);
        }
    }

#if MASKED_GFNI_TABLE2
    i = validate_gfni_primitives(&n[0].Mat, &e_prime_inv[0][0]);
    if (i != MASKED_WBSM4_ENHANCED_OK) return i;
#endif

    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        uint32_t prime_constant = cus_random();
        uint32_t double_prime_constant = p[i + 4].Vec.V ^ prime_constant;
        Aff32 p_prime = make_aff32(p[i + 4].Mat, prime_constant);
        Aff32 p_double_prime =
            make_aff32(p[i + 4].Mat, double_prime_constant);
        Aff32 e_prime32;
        Aff32 mask32;

        affinecomM8to32(e_prime[i][0], e_prime[i][1],
                        e_prime[i][2], e_prime[i][3], &e_prime32);
        affinecomM8to32(mask_encoding[i], mask_encoding[i],
                        mask_encoding[i], mask_encoding[i], &mask32);

        ctx->masked_data[i] = compose_aff32(
            p_prime, compose_aff32(linear_affine, e_prime32));
        ctx->mask_branch[i] = compose_aff32(
            n32_inv, compose_aff32(linear_affine, mask32));
        ctx->round_x[i] = compose_aff32(p_double_prime, p_inv[i]);
    }

    for (i = 0; i < 4; ++i) {
        ctx->input_encoding[i] = p[i];
        ctx->output_decoding[i] = p_inv[35 - i];
    }
    prepare_simd_context(ctx);

    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        uint32_t round_key = (uint32_t)key_schedule.sk[i];

        for (j = 0; j < 4; ++j) {
            uint8_t round_key_byte =
                (uint8_t)(round_key >> (24 - 8 * j));

            for (x = 0; x < 256; ++x) {
                uint8_t y = (uint8_t)x;
                uint8_t decoded_y = affineU8(e[i][j], y);
                uint8_t mask = recover_round_mask(
                    ctx, mask_encoding, i, j, y);

                if (i == 0) {
                    uint8_t value =
                        SBOX[decoded_y ^ round_key_byte] ^ mask;
                    ctx->table1[j][x] =
                        affineU8(e_prime_inv[i][j], value);
                } else {
                    int previous;

#if MASKED_GFNI_TABLE2
                    const __m512i sequence = _mm512_loadu_si512(
                        (const void *)GFNI_SEQUENCE);
                    for (previous = 0; previous < 256; previous += 64) {
                        __m512i previous_values = _mm512_add_epi8(
                            sequence, _mm512_set1_epi8((char)previous));
                        __m512i previous_masks = matmul_m8_zmm(
                            &n[j].Mat, previous_values);
                        __m512i substituted = sm4_sbox_zmm(_mm512_xor_si512(
                            previous_masks,
                            _mm512_set1_epi8((char)(decoded_y ^
                                                    round_key_byte))));
                        __m512i encoded = affine_u8_zmm(
                            &e_prime_inv[i][j],
                            _mm512_xor_si512(
                                substituted, _mm512_set1_epi8((char)mask)));
                        const uint16_t index = (uint16_t)(
                            ((uint16_t)y << 8) | (uint16_t)previous);

                        _mm512_storeu_si512(
                            (void *)(ctx->table2 +
                                table2_offset(i, j, index)), encoded);
#if MASKED_GFNI_VERIFY_TABLE2
                        {
                            int verify;
                            for (verify = 0; verify < 64; ++verify) {
                                const uint8_t previous_value =
                                    (uint8_t)(previous + verify);
                                const uint8_t previous_mask = MatMulNumM8(
                                    n[j].Mat, previous_value);
                                const uint8_t value =
                                    SBOX[decoded_y ^ previous_mask ^
                                         round_key_byte] ^ mask;
                                const uint8_t expected = affineU8(
                                    e_prime_inv[i][j], value);
                                if (ctx->table2[table2_offset(
                                        i, j,
                                        (uint16_t)(index + verify))] !=
                                    expected) {
                                    return MASKED_WBSM4_ENHANCED_GFNI_TABLE_MISMATCH;
                                }
                            }
                        }
#endif
                    }
#else
                    for (previous = 0; previous < 256; ++previous) {
                        uint8_t previous_mask = MatMulNumM8(
                            n[j].Mat, (uint8_t)previous);
                        uint8_t value =
                            SBOX[decoded_y ^ previous_mask ^
                                 round_key_byte] ^ mask;
                        uint16_t index =
                            (uint16_t)(((uint16_t)y << 8) |
                                       (uint16_t)previous);

                        ctx->table2[table2_offset(i, j, index)] =
                            affineU8(e_prime_inv[i][j], value);
                    }
#endif
                }
            }
        }
    }

#if MASKED_PRECOMPOSE_MASK_BRANCH
    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        for (j = 0; j < 4; ++j) {
            for (x = 0; x < 256; x += 16) {
                uint32_t words[16];
                __m512i contribution;
                int lane;

                for (lane = 0; lane < 16; ++lane) {
                    const uint8_t index = phi_power(
                        ctx->phi[i], (uint8_t)(x + lane), j);
                    const uint32_t value = ctx->mask_table[i][index];
                    words[lane] = j == 3
                        ? value : value << (24 - 8 * j);
                }
                contribution = apply_aff32_16(
                    &ctx->prepared_mask_branch[i],
                    _mm512_loadu_si512((const void *)words));
                if (j != 0) {
                    contribution = _mm512_xor_si512(
                        contribution,
                        _mm512_set1_epi32((int)ctx->mask_branch[i].Vec.V));
                }
                _mm512_storeu_si512(
                    (void *)&ctx->simd_mask_branch[i][j][x],
                    contribution);
#if MASKED_GFNI_VERIFY_MASK_BRANCH
                for (lane = 0; lane < 16; ++lane) {
                    uint32_t expected = affineU32(
                        ctx->mask_branch[i], words[lane]);
                    if (j != 0) {
                        expected ^= ctx->mask_branch[i].Vec.V;
                    }
                    if (ctx->simd_mask_branch[i][j][x + lane] != expected) {
                        return MASKED_WBSM4_ENHANCED_GFNI_MASK_BRANCH_MISMATCH;
                    }
                }
#endif
            }
        }
    }
#endif
#if MASKED_PRECOMPOSE_MASK_LOOKUP_U8
    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        for (j = 0; j < 4; ++j) {
            for (x = 0; x < 256; ++x) {
                const uint8_t index = phi_power(
                    ctx->phi[i], (uint8_t)x, j);
                ctx->simd_mask_lookup_u8[i][j][x] =
                    ctx->mask_table[i][index];
            }
        }
    }
#endif

    memset(&key_schedule, 0, sizeof(key_schedule));
    ctx->magic = MASKED_WBSM4_ENHANCED_MAGIC;
    return masked_wbsm4_enhanced_validate(ctx);
}

static uint32_t lookup_masked_sbox(
    const masked_wbsm4_enhanced_context *ctx,
    int round,
    uint32_t y,
    uint32_t previous_masks)
{
    uint32_t result = 0;
    int j;

    for (j = 0; j < 4; ++j) {
        uint8_t y_byte = (uint8_t)(y >> (24 - 8 * j));
        uint8_t value;

        if (round == 0) {
            value = ctx->table1[j][y_byte];
        } else {
            uint8_t previous =
                (uint8_t)(previous_masks >> (24 - 8 * j));
            uint16_t index =
                (uint16_t)(((uint16_t)y_byte << 8) | previous);
            value = ctx->table2[table2_offset(round, j, index)];
        }
        result |= (uint32_t)value << (24 - 8 * j);
    }
    return result;
}

static uint32_t lookup_mask_value(
    const masked_wbsm4_enhanced_context *ctx,
    int round,
    uint32_t y)
{
    uint32_t result = 0;
    int j;

    for (j = 0; j < 4; ++j) {
        uint8_t y_byte = (uint8_t)(y >> (24 - 8 * j));
        uint8_t index = phi_power(ctx->phi[round], y_byte, j);
        result |= (uint32_t)ctx->mask_table[round][index]
                  << (24 - 8 * j);
    }
    return result;
}

static __m512i lookup_masked_sbox16(
    const masked_wbsm4_enhanced_context *ctx,
    int round,
    __m512i y,
    __m512i previous_masks)
{
    __m512i result = _mm512_setzero_si512();
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
        const __m512i y_byte = extract_byte32(y, byte_index);
        __m512i value;

        if (round == 0) {
            value = gather_u8_16(ctx->table1[byte_index], y_byte);
        } else {
            const __m512i previous = extract_byte32(
                previous_masks, byte_index);
            const __m512i index = _mm512_or_si512(
                _mm512_slli_epi32(y_byte, 8), previous);
            const uint8_t *table = ctx->table2 +
                ((size_t)(round - 1) * 4u + (size_t)byte_index) *
                    TABLE2_ENTRIES;
            value = gather_u8_16(table, index);
        }
        result = insert_byte32(result, value, byte_index);
    }
    return result;
}

#if !MASKED_PRECOMPOSE_MASK_BRANCH
static __m512i lookup_mask_value16(
    const masked_wbsm4_enhanced_context *ctx,
    int round,
    __m512i y)
{
    __m512i result = _mm512_setzero_si512();
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
#if MASKED_PRECOMPOSE_MASK_LOOKUP_U8
        const __m512i index = extract_byte32(y, byte_index);
        result = insert_byte32(
            result,
            gather_u8_16(
                ctx->simd_mask_lookup_u8[round][byte_index], index),
            byte_index);
#else
        __m512i index = extract_byte32(y, byte_index);
        int power;

        for (power = 0; power < byte_index; ++power) {
            index = gather_u8_16(ctx->phi[round], index);
        }
        result = insert_byte32(
            result, gather_u8_16(ctx->mask_table[round], index), byte_index);
#endif
    }
    return result;
}

#if MASKED_FUSE_U8_MASK_AFFINE
static __m512i lookup_mask_branch_u8_16(
    const masked_wbsm4_enhanced_context *ctx,
    int round,
    __m512i y)
{
    __m128i input_bytes[4];
    __m128i output_bytes[4];
    const prepared_aff32 *affine = &ctx->prepared_mask_branch[round];
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
        input_bytes[byte_index] = _mm512_cvtepi32_epi8(
            gather_u8_16(
                ctx->simd_mask_lookup_u8[round][byte_index],
                extract_byte32(y, byte_index)));
    }
    for (byte_index = 0; byte_index < 4; ++byte_index) {
        output_bytes[byte_index] = _mm_xor_si128(
            apply_linear_byte16(affine, input_bytes, byte_index),
            _mm_set1_epi8((char)affine->vector[byte_index]));
    }
    return join_bytes16(output_bytes);
}
#endif
#endif

#if MASKED_PRECOMPOSE_MASK_BRANCH
static __m512i lookup_mask_branch16(
    const masked_wbsm4_enhanced_context *ctx,
    int round,
    __m512i y)
{
    __m512i result = _mm512_setzero_si512();
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
        result = _mm512_xor_si512(
            result,
            _mm512_i32gather_epi32(
                extract_byte32(y, byte_index),
                (const void *)ctx->simd_mask_branch[round][byte_index],
                4));
    }
    return result;
}
#endif

static void encrypt16(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t input[16 * 16],
    uint8_t output[16 * 16])
{
    __m512i state[4];
    __m512i state_masks[4] = {
        _mm512_setzero_si512(), _mm512_setzero_si512(),
        _mm512_setzero_si512(), _mm512_setzero_si512()
    };
    int round;
    int word;

    for (word = 0; word < 4; ++word) {
        state[word] = apply_aff32_16(
            &ctx->prepared_input_encoding[word], load_word16(input, word));
    }

    for (round = 0; round < ENHANCED_ROUNDS; ++round) {
        const int x0_slot = round & 3;
        const int x1_slot = (round + 1) & 3;
        const int x2_slot = (round + 2) & 3;
        const int x3_slot = (round + 3) & 3;
        const __m512i y = apply_aff32_xor3_16(
            &ctx->prepared_part1[round][0], state[x1_slot],
            &ctx->prepared_part1[round][1], state[x2_slot],
            &ctx->prepared_part1[round][2], state[x3_slot]);
        const __m512i previous_masks = _mm512_xor_si512(
            _mm512_xor_si512(state_masks[x1_slot], state_masks[x2_slot]),
            state_masks[x3_slot]);
        const __m512i encoded_sbox = lookup_masked_sbox16(
            ctx, round, y, previous_masks);
#if MASKED_PRECOMPOSE_MASK_BRANCH
        const __m512i new_mask_branch = lookup_mask_branch16(
            ctx, round, y);
#elif MASKED_FUSE_U8_MASK_AFFINE
        const __m512i new_mask_branch = lookup_mask_branch_u8_16(
            ctx, round, y);
#else
        const __m512i encoded_round_masks = lookup_mask_value16(
            ctx, round, y);
        const __m512i new_mask_branch = apply_aff32_16(
            &ctx->prepared_mask_branch[round], encoded_round_masks);
#endif

        state[x0_slot] = apply_aff32_xor2_16(
            &ctx->prepared_round_x[round], state[x0_slot],
            &ctx->prepared_masked_data[round], encoded_sbox);
        state_masks[x0_slot] = _mm512_xor_si512(
            state_masks[x0_slot], new_mask_branch);
    }

    for (word = 0; word < 4; ++word) {
        const int state_slot = 3 - word;
        const __m512i decoded = apply_aff32_xor2_16(
            &ctx->prepared_output_decoding[word], state[state_slot],
            &ctx->prepared_mask_decode, state_masks[state_slot]);
        store_word16(output, word, decoded);
    }
}

#if MASKED_PIPELINE_32WAY
static void encrypt32(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t input[32 * 16],
    uint8_t output[32 * 16])
{
    __m512i ax0 = apply_aff32_16(
        &ctx->prepared_input_encoding[0], load_word16(input, 0));
    __m512i ax1 = apply_aff32_16(
        &ctx->prepared_input_encoding[1], load_word16(input, 1));
    __m512i ax2 = apply_aff32_16(
        &ctx->prepared_input_encoding[2], load_word16(input, 2));
    __m512i ax3 = apply_aff32_16(
        &ctx->prepared_input_encoding[3], load_word16(input, 3));
    __m512i bx0 = apply_aff32_16(
        &ctx->prepared_input_encoding[0], load_word16(input + 256, 0));
    __m512i bx1 = apply_aff32_16(
        &ctx->prepared_input_encoding[1], load_word16(input + 256, 1));
    __m512i bx2 = apply_aff32_16(
        &ctx->prepared_input_encoding[2], load_word16(input + 256, 2));
    __m512i bx3 = apply_aff32_16(
        &ctx->prepared_input_encoding[3], load_word16(input + 256, 3));
    __m512i am0 = _mm512_setzero_si512();
    __m512i am1 = _mm512_setzero_si512();
    __m512i am2 = _mm512_setzero_si512();
    __m512i am3 = _mm512_setzero_si512();
    __m512i bm0 = _mm512_setzero_si512();
    __m512i bm1 = _mm512_setzero_si512();
    __m512i bm2 = _mm512_setzero_si512();
    __m512i bm3 = _mm512_setzero_si512();
    int round;

    for (round = 0; round < ENHANCED_ROUNDS; ++round) {
        __m512i y = apply_aff32_xor3_16(
            &ctx->prepared_part1[round][0], ax1,
            &ctx->prepared_part1[round][1], ax2,
            &ctx->prepared_part1[round][2], ax3);
        __m512i previous_masks = _mm512_xor_si512(
            _mm512_xor_si512(am1, am2), am3);
        __m512i encoded_sbox = lookup_masked_sbox16(
            ctx, round, y, previous_masks);
        __m512i new_mask_branch;
        __m512i new_state;
        __m512i new_state_mask;
#if MASKED_PRECOMPOSE_MASK_BRANCH
        new_mask_branch = lookup_mask_branch16(ctx, round, y);
#elif MASKED_FUSE_U8_MASK_AFFINE
        new_mask_branch = lookup_mask_branch_u8_16(ctx, round, y);
#else
        new_mask_branch = apply_aff32_16(
            &ctx->prepared_mask_branch[round],
            lookup_mask_value16(ctx, round, y));
#endif
        new_state = apply_aff32_xor2_16(
            &ctx->prepared_round_x[round], ax0,
            &ctx->prepared_masked_data[round], encoded_sbox);
        new_state_mask = _mm512_xor_si512(am0, new_mask_branch);
        ax0 = ax1;
        ax1 = ax2;
        ax2 = ax3;
        ax3 = new_state;
        am0 = am1;
        am1 = am2;
        am2 = am3;
        am3 = new_state_mask;

        y = apply_aff32_xor3_16(
            &ctx->prepared_part1[round][0], bx1,
            &ctx->prepared_part1[round][1], bx2,
            &ctx->prepared_part1[round][2], bx3);
        previous_masks = _mm512_xor_si512(
            _mm512_xor_si512(bm1, bm2), bm3);
        encoded_sbox = lookup_masked_sbox16(
            ctx, round, y, previous_masks);
#if MASKED_PRECOMPOSE_MASK_BRANCH
        new_mask_branch = lookup_mask_branch16(ctx, round, y);
#elif MASKED_FUSE_U8_MASK_AFFINE
        new_mask_branch = lookup_mask_branch_u8_16(ctx, round, y);
#else
        new_mask_branch = apply_aff32_16(
            &ctx->prepared_mask_branch[round],
            lookup_mask_value16(ctx, round, y));
#endif
        new_state = apply_aff32_xor2_16(
            &ctx->prepared_round_x[round], bx0,
            &ctx->prepared_masked_data[round], encoded_sbox);
        new_state_mask = _mm512_xor_si512(bm0, new_mask_branch);
        bx0 = bx1;
        bx1 = bx2;
        bx2 = bx3;
        bx3 = new_state;
        bm0 = bm1;
        bm1 = bm2;
        bm2 = bm3;
        bm3 = new_state_mask;
    }

    store_word16(output, 0, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[0], ax3,
        &ctx->prepared_mask_decode, am3));
    store_word16(output, 1, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[1], ax2,
        &ctx->prepared_mask_decode, am2));
    store_word16(output, 2, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[2], ax1,
        &ctx->prepared_mask_decode, am1));
    store_word16(output, 3, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[3], ax0,
        &ctx->prepared_mask_decode, am0));
    store_word16(output + 256, 0, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[0], bx3,
        &ctx->prepared_mask_decode, bm3));
    store_word16(output + 256, 1, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[1], bx2,
        &ctx->prepared_mask_decode, bm2));
    store_word16(output + 256, 2, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[2], bx1,
        &ctx->prepared_mask_decode, bm1));
    store_word16(output + 256, 3, apply_aff32_xor2_16(
        &ctx->prepared_output_decoding[3], bx0,
        &ctx->prepared_mask_decode, bm0));
}
#endif

static void encrypt_encoded(
    const masked_wbsm4_enhanced_context *ctx,
    const uint32_t input[4],
    uint32_t output[4],
    uint32_t output_masks[4])
{
    uint32_t state[4] = {input[0], input[1], input[2], input[3]};
    uint32_t state_masks[4] = {0, 0, 0, 0};
    int i;

    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        int x0_slot = i & 3;
        int x1_slot = (i + 1) & 3;
        int x2_slot = (i + 2) & 3;
        int x3_slot = (i + 3) & 3;
        uint32_t y =
            affineU32(ctx->part1[i][0], state[x1_slot]) ^
            affineU32(ctx->part1[i][1], state[x2_slot]) ^
            affineU32(ctx->part1[i][2], state[x3_slot]);
        uint32_t previous_masks =
            state_masks[x1_slot] ^ state_masks[x2_slot] ^
            state_masks[x3_slot];
        uint32_t encoded_sbox =
            lookup_masked_sbox(ctx, i, y, previous_masks);
        uint32_t encoded_round_masks = lookup_mask_value(ctx, i, y);
        uint32_t data_branch =
            affineU32(ctx->masked_data[i], encoded_sbox);
        uint32_t new_mask_branch =
            affineU32(ctx->mask_branch[i], encoded_round_masks);

        state[x0_slot] =
            affineU32(ctx->round_x[i], state[x0_slot]) ^ data_branch;
        state_masks[x0_slot] ^= new_mask_branch;
    }

    output[0] = state[3];
    output[1] = state[2];
    output[2] = state[1];
    output[3] = state[0];
    output_masks[0] = state_masks[3];
    output_masks[1] = state_masks[2];
    output_masks[2] = state_masks[1];
    output_masks[3] = state_masks[0];
}

int masked_wbsm4_enhanced_encrypt(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t input[16],
    uint8_t output[16])
{
    uint32_t encoded_input[4];
    uint32_t encoded_output[4];
    uint32_t encoded_output_masks[4];
    int i;

    if (ctx == NULL || input == NULL || output == NULL) {
        return MASKED_WBSM4_ENHANCED_BAD_ARGUMENT;
    }
    if (ctx->magic != MASKED_WBSM4_ENHANCED_MAGIC) {
        return MASKED_WBSM4_ENHANCED_NOT_GENERATED;
    }

    for (i = 0; i < 4; ++i) {
        encoded_input[i] = affineU32(
            ctx->input_encoding[i], load_be32(input + 4 * i));
    }
    encrypt_encoded(ctx, encoded_input, encoded_output,
                    encoded_output_masks);

    for (i = 0; i < 4; ++i) {
        uint32_t decoded = affineU32(
            ctx->output_decoding[i], encoded_output[i]);
        decoded ^= MatMulNumM32(ctx->mask_decode,
                               encoded_output_masks[i]);
        store_be32(decoded, output + 4 * i);
    }
    return MASKED_WBSM4_ENHANCED_OK;
}

int masked_wbsm4_enhanced_encrypt_many_scalar(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t *input,
    uint8_t *output,
    size_t block_count)
{
    size_t block;

    if (ctx == NULL || input == NULL || output == NULL) {
        return MASKED_WBSM4_ENHANCED_BAD_ARGUMENT;
    }
    for (block = 0; block < block_count; ++block) {
        const int status = masked_wbsm4_enhanced_encrypt(
            ctx, input + block * 16, output + block * 16);
        if (status != MASKED_WBSM4_ENHANCED_OK) return status;
    }
    return MASKED_WBSM4_ENHANCED_OK;
}

int masked_wbsm4_enhanced_encrypt_many(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t *input,
    uint8_t *output,
    size_t block_count)
{
    if (ctx == NULL || input == NULL || output == NULL) {
        return MASKED_WBSM4_ENHANCED_BAD_ARGUMENT;
    }
    if (ctx->magic != MASKED_WBSM4_ENHANCED_MAGIC) {
        return MASKED_WBSM4_ENHANCED_NOT_GENERATED;
    }
#if MASKED_PIPELINE_32WAY
    while (block_count >= 32) {
        encrypt32(ctx, input, output);
        input += 32 * 16;
        output += 32 * 16;
        block_count -= 32;
    }
#endif
    while (block_count >= 16) {
        encrypt16(ctx, input, output);
        input += 16 * 16;
        output += 16 * 16;
        block_count -= 16;
    }
    return masked_wbsm4_enhanced_encrypt_many_scalar(
        ctx, input, output, block_count);
}

int masked_wbsm4_enhanced_validate(
    const masked_wbsm4_enhanced_context *ctx)
{
    int i;

    if (ctx == NULL || ctx->table2 == NULL) {
        return MASKED_WBSM4_ENHANCED_BAD_ARGUMENT;
    }
    if (ctx->magic != MASKED_WBSM4_ENHANCED_MAGIC) {
        return MASKED_WBSM4_ENHANCED_NOT_GENERATED;
    }
    for (i = 0; i < ENHANCED_ROUNDS; ++i) {
        if (!permutation_is_valid(ctx->phi[i])) {
            return MASKED_WBSM4_ENHANCED_INVALID_TABLES;
        }
    }
    return MASKED_WBSM4_ENHANCED_OK;
}

size_t masked_wbsm4_enhanced_lookup_table_bytes(void)
{
    size_t bytes = sizeof(((masked_wbsm4_enhanced_context *)0)->table1) +
           TABLE2_BYTES +
           sizeof(((masked_wbsm4_enhanced_context *)0)->mask_table) +
           sizeof(((masked_wbsm4_enhanced_context *)0)->phi);
#if MASKED_PRECOMPOSE_MASK_BRANCH
    bytes += sizeof(
        ((masked_wbsm4_enhanced_context *)0)->simd_mask_branch);
#endif
#if MASKED_PRECOMPOSE_MASK_LOOKUP_U8
    bytes += sizeof(
        ((masked_wbsm4_enhanced_context *)0)->simd_mask_lookup_u8);
#endif
    return bytes;
}

size_t masked_wbsm4_enhanced_allocated_bytes(
    const masked_wbsm4_enhanced_context *ctx)
{
    return ctx == NULL ? 0 : sizeof(*ctx) + TABLE2_BYTES;
}

int masked_wbsm4_enhanced_generate(masked_wbsm4_enhanced_context *ctx,
                                   const uint8_t key[16],
                                   uint32_t random_seed)
{
    return masked_wbsm4_enhanced_generate_direction(ctx, key, random_seed, 0);
}
