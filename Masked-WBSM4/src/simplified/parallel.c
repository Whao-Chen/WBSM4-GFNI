#include "masked_wbsm4.h"

#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"
#include "sbox.h"
#include "sm4.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define MASKED_ROUNDS 8
#define TABLE1_ROUNDS 2
#define TABLE2_ROUNDS 6
#define TABLE3_ROUNDS 24
#define TABLE2_ENTRIES 65536u
#define TABLE2_BYTES ((size_t)TABLE2_ROUNDS * 4u * TABLE2_ENTRIES)
#define MASKED_WBSM4_MAGIC UINT32_C(0x4d574234)

#ifndef MASKED_GFNI_TABLE2
#define MASKED_GFNI_TABLE2 1
#endif
#ifndef MASKED_GFNI_VERIFY_TABLE2
#define MASKED_GFNI_VERIFY_TABLE2 0
#endif

typedef struct {
    uint64_t matrix[4][4];
    uint8_t vector[4];
} prepared_aff32;

struct masked_wbsm4_context {
    Aff32 part1[32][3];
    Aff32 round_x[32];
    Aff32 round_z[32];
    Aff32 masked_data[MASKED_ROUNDS];
    Aff32 mask_branch[MASKED_ROUNDS];
    M32 unload_part1[3];
    Aff32 unload_state[4];
    Aff32 input_encoding[4];
    Aff32 output_decoding[4];

    uint8_t table1[TABLE1_ROUNDS][4][256];
    uint8_t *table2;
    uint32_t table3[TABLE3_ROUNDS][4][256];
    uint8_t mask_table[MASKED_ROUNDS][256];
    uint8_t phi[MASKED_ROUNDS][256];

    prepared_aff32 prepared_part1[32][3];
    prepared_aff32 prepared_round_x[32];
    prepared_aff32 prepared_round_z[32];
    prepared_aff32 prepared_masked_data[MASKED_ROUNDS];
    prepared_aff32 prepared_mask_branch[MASKED_ROUNDS];
    prepared_aff32 prepared_unload_part1[3];
    prepared_aff32 prepared_unload_state[4];
    prepared_aff32 prepared_input_encoding[4];
    prepared_aff32 prepared_output_decoding[4];

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

static uint8_t backend_affine_u8(const Aff8 *affine, uint8_t value);
static uint8_t backend_matmul_num_m8(const M8 *matrix, uint8_t value);

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
        if (output[value] != SBOX[input[value]]) return -10;
    }

    _mm512_storeu_si512(
        (void *)output,
        matmul_m8_zmm(matrix, _mm512_loadu_si512((const void *)input)));
    for (value = 0; value < 64; ++value) {
        if (output[value] != backend_matmul_num_m8(matrix, input[value])) {
            return -11;
        }
    }

    _mm512_storeu_si512(
        (void *)output,
        affine_u8_zmm(affine, _mm512_loadu_si512((const void *)input)));
    for (value = 0; value < 64; ++value) {
        if (output[value] != backend_affine_u8(affine, input[value])) {
            return -12;
        }
    }
    return 0;
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
    affinemixM32_final_matmulmat_matmulnum_wrapper_ptr(
        &outer, &inner, &result);
    return result;
}

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

static void identity_aff8(Aff8 *affine)
{
    identityM8(&affine->Mat);
    affine->Vec.V = 0;
}

static int masked_slot(int round)
{
    return round < 4 ? round : round - 24;
}

static int table2_slot(int round)
{
    return round < 4 ? round - 1 : round - 26;
}

static uint8_t phi_power(const uint8_t phi[256], uint8_t value, int power)
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
    int value;
    int phi2_identity = 1;
    int phi3_identity = 1;

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

static uint8_t recover_mask(const masked_wbsm4_context *ctx,
                            const Aff8 mask_encoding[MASKED_ROUNDS],
                            int slot,
                            int byte_index,
                            uint8_t table_input)
{
    uint8_t index = phi_power(ctx->phi[slot], table_input, byte_index);
    uint8_t encoded = ctx->mask_table[slot][index];
    return slot < 4 ? backend_affine_u8(&mask_encoding[slot], encoded) : encoded;
}

static size_t table2_offset(int slot, int byte_index, uint16_t input)
{
    return (((size_t)slot * 4u + (size_t)byte_index) * TABLE2_ENTRIES) + input;
}

static void prepare_simd_context(masked_wbsm4_context *ctx)
{
    int round;

    for (round = 0; round < 32; ++round) {
        int word;
        for (word = 0; word < 3; ++word) {
            prepare_aff32(&ctx->part1[round][word],
                          &ctx->prepared_part1[round][word]);
        }
        prepare_aff32(&ctx->round_x[round], &ctx->prepared_round_x[round]);
        prepare_aff32(&ctx->round_z[round], &ctx->prepared_round_z[round]);
    }
    for (round = 0; round < MASKED_ROUNDS; ++round) {
        prepare_aff32(&ctx->masked_data[round],
                      &ctx->prepared_masked_data[round]);
        prepare_aff32(&ctx->mask_branch[round],
                      &ctx->prepared_mask_branch[round]);
    }
    for (round = 0; round < 3; ++round) {
        prepare_matrix32(&ctx->unload_part1[round],
                         &ctx->prepared_unload_part1[round]);
    }
    for (round = 0; round < 4; ++round) {
        prepare_aff32(&ctx->unload_state[round],
                      &ctx->prepared_unload_state[round]);
        prepare_aff32(&ctx->input_encoding[round],
                      &ctx->prepared_input_encoding[round]);
        prepare_aff32(&ctx->output_decoding[round],
                      &ctx->prepared_output_decoding[round]);
    }
}

masked_wbsm4_context *masked_wbsm4_create(void)
{
    masked_wbsm4_context *ctx = calloc(1, sizeof(*ctx));
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

void masked_wbsm4_destroy(masked_wbsm4_context *ctx)
{
    if (ctx != NULL) {
        free(ctx->table2);
        memset(ctx, 0, sizeof(*ctx));
        free(ctx);
    }
}

int masked_wbsm4_generate_direction(masked_wbsm4_context *ctx,
                          const uint8_t key[16],
                          uint32_t random_seed, int decrypt)
{
    Aff32 p[36];
    Aff32 p_inv[36];
    Aff8 e[32][4];
    Aff8 e_inv[32][4];
    Aff32 e32_inv[32];
    Aff8 e_prime[MASKED_ROUNDS][4];
    Aff8 e_prime_inv[MASKED_ROUNDS][4];
    Aff8 mask_encoding[MASKED_ROUNDS];
    Aff8 mask_encoding_inv[MASKED_ROUNDS];
    Aff8 n[4];
    Aff8 n_inv[4];
    Aff32 n32;
    Aff32 n32_inv;
    Aff32 q[32];
    Aff32 q_inv[32];
    M32 linear_matrix;
    Aff32 linear_affine;
    sm4_context key_schedule;
    uint8_t *table2;
    int i;
    int j;
    int x;

    if (ctx == NULL || key == NULL || ctx->table2 == NULL) {
        return MASKED_WBSM4_BAD_ARGUMENT;
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
        backend_gen_affinepair_m32(&p[i], &p_inv[i]);
    }

    for (i = 0; i < 32; ++i) {
        for (j = 0; j < 4; ++j) {
            backend_gen_affinepair_m8(&e[i][j], &e_inv[i][j]);
        }
        affinecomM8to32(e_inv[i][0], e_inv[i][1],
                        e_inv[i][2], e_inv[i][3], &e32_inv[i]);

        for (j = 0; j < 3; ++j) {
            ctx->part1[i][j] = compose_aff32(e32_inv[i], p_inv[i + j + 1]);
        }
    }

    for (j = 0; j < 4; ++j) {
        backend_gen_matpair_m8(&n[j].Mat, &n_inv[j].Mat);
        n[j].Vec.V = 0;
        n_inv[j].Vec.V = 0;
    }
    affinecomM8to32(n[0], n[1], n[2], n[3], &n32);
    affinecomM8to32(n_inv[0], n_inv[1], n_inv[2], n_inv[3], &n32_inv);

    for (i = 0; i < MASKED_ROUNDS; ++i) {
        for (j = 0; j < 4; ++j) {
            backend_gen_affinepair_m8(&e_prime[i][j], &e_prime_inv[i][j]);
        }
        if (i < 4) {
            backend_gen_affinepair_m8(&mask_encoding[i], &mask_encoding_inv[i]);
        } else {
            identity_aff8(&mask_encoding[i]);
            identity_aff8(&mask_encoding_inv[i]);
        }
        generate_permutation(ctx->phi[i]);
        for (x = 0; x < 256; ++x) {
            uint8_t mask = (uint8_t)cus_random();
            ctx->mask_table[i][x] = i < 4
                ? backend_affine_u8(&mask_encoding_inv[i], mask)
                : mask;
        }
    }

#if MASKED_GFNI_TABLE2
    i = validate_gfni_primitives(&n[0].Mat, &e_prime_inv[0][0]);
    if (i != 0) return i;
#endif

    for (i = 4; i <= 27; ++i) {
        backend_gen_affinepair_m32(&q[i], &q_inv[i]);
    }

    for (i = 0; i < 32; ++i) {
        uint32_t prime_constant = cus_random();
        uint32_t double_prime_constant = p[i + 4].Vec.V ^ prime_constant;

        if (i < 4 || i >= 28) {
            int slot = masked_slot(i);
            Aff32 p_prime = make_aff32(p[i + 4].Mat, prime_constant);
            Aff32 p_double_prime = make_aff32(p[i + 4].Mat,
                                              double_prime_constant);
            Aff32 e_prime32;
            Aff32 mask32;

            affinecomM8to32(e_prime[slot][0], e_prime[slot][1],
                            e_prime[slot][2], e_prime[slot][3], &e_prime32);
            ctx->masked_data[slot] = compose_aff32(
                p_prime, compose_aff32(linear_affine, e_prime32));
            ctx->round_x[i] = compose_aff32(p_double_prime, p_inv[i]);

            if (slot < 4) {
                affinecomM8to32(mask_encoding[slot], mask_encoding[slot],
                                mask_encoding[slot], mask_encoding[slot], &mask32);
                ctx->mask_branch[slot] = compose_aff32(
                    n32_inv, compose_aff32(linear_affine, mask32));
            } else {
                ctx->mask_branch[slot] = linear_affine;
            }
        } else if (i <= 7) {
            Aff32 x_transform = compose_aff32(p[i + 4], p_inv[i]);
            Aff32 z_outer = make_aff32(p[i + 4].Mat, prime_constant);
            M32 unload_part1;
            M32 unload_state;

            x_transform.Vec.V ^= p[i + 4].Vec.V;
            ctx->round_x[i] = x_transform;
            ctx->round_z[i] = compose_aff32(z_outer, q_inv[i]);

            if (i <= 6) {
                backend_matmul_mat_m32(&e32_inv[i].Mat, &n32.Mat,
                                          &unload_part1);
                ctx->unload_part1[i - 4] = unload_part1;
            }
            backend_matmul_mat_m32(&p[i + 4].Mat, &n32.Mat,
                                      &unload_state);
            ctx->unload_state[i - 4] = make_aff32(unload_state,
                                                   double_prime_constant);
        } else {
            Aff32 p_prime = make_aff32(p[i + 4].Mat, prime_constant);
            Aff32 p_double_prime = make_aff32(p[i + 4].Mat,
                                              double_prime_constant);
            ctx->round_x[i] = compose_aff32(p_double_prime, p_inv[i]);
            ctx->round_z[i] = compose_aff32(p_prime, q_inv[i]);
        }
    }

    for (i = 0; i < 4; ++i) {
        ctx->input_encoding[i] = p[i];
        ctx->output_decoding[i] = p_inv[35 - i];
    }

    for (i = 0; i < 32; ++i) {
        uint32_t round_key = (uint32_t)key_schedule.sk[i];

        if (i == 0 || i == 28) {
            int table_slot = i == 0 ? 0 : 1;
            int slot = masked_slot(i);
            for (j = 0; j < 4; ++j) {
                uint8_t round_key_byte = (uint8_t)(round_key >> (24 - 8 * j));
                for (x = 0; x < 256; ++x) {
                    uint8_t value = backend_affine_u8(&e[i][j], (uint8_t)x);
                    uint8_t mask = recover_mask(ctx, mask_encoding, slot, j,
                                                (uint8_t)x);
                    value = SBOX[value ^ round_key_byte] ^ mask;
                    ctx->table1[table_slot][j][x] =
                        backend_affine_u8(&e_prime_inv[slot][j], value);
                }
            }
        } else if ((i >= 1 && i <= 3) || (i >= 29 && i <= 31)) {
            int slot = masked_slot(i);
            int t2_slot = table2_slot(i);
            for (j = 0; j < 4; ++j) {
                uint8_t round_key_byte = (uint8_t)(round_key >> (24 - 8 * j));
                for (x = 0; x < 256; ++x) {
                    uint8_t y = (uint8_t)x;
                    uint8_t decoded_y = backend_affine_u8(&e[i][j], y);
                    uint8_t mask = recover_mask(ctx, mask_encoding, slot, j, y);
                    int previous;
#if MASKED_GFNI_TABLE2
                    const __m512i sequence = _mm512_loadu_si512(
                        (const void *)GFNI_SEQUENCE);
                    for (previous = 0; previous < 256; previous += 64) {
                        __m512i previous_values = _mm512_add_epi8(
                            sequence, _mm512_set1_epi8((char)previous));
                        __m512i previous_masks = slot < 4
                            ? matmul_m8_zmm(&n[j].Mat, previous_values)
                            : previous_values;
                        __m512i substituted = sm4_sbox_zmm(_mm512_xor_si512(
                            previous_masks,
                            _mm512_set1_epi8((char)(decoded_y ^
                                                    round_key_byte))));
                        __m512i encoded = affine_u8_zmm(
                            &e_prime_inv[slot][j],
                            _mm512_xor_si512(
                                substituted, _mm512_set1_epi8((char)mask)));
                        const uint16_t index = (uint16_t)(
                            ((uint16_t)y << 8) | (uint16_t)previous);
                        _mm512_storeu_si512(
                            (void *)(ctx->table2 +
                                table2_offset(t2_slot, j, index)), encoded);
#if MASKED_GFNI_VERIFY_TABLE2
                        {
                            int verify;
                            for (verify = 0; verify < 64; ++verify) {
                                const uint8_t previous_value =
                                    (uint8_t)(previous + verify);
                                const uint8_t previous_mask = slot < 4
                                    ? backend_matmul_num_m8(
                                        &n[j].Mat, previous_value)
                                    : previous_value;
                                const uint8_t value =
                                    SBOX[decoded_y ^ previous_mask ^
                                         round_key_byte] ^ mask;
                                const uint8_t expected = backend_affine_u8(
                                    &e_prime_inv[slot][j], value);
                                if (ctx->table2[table2_offset(
                                        t2_slot, j,
                                        (uint16_t)(index + verify))] !=
                                    expected) {
                                    return -20;
                                }
                            }
                        }
#endif
                    }
#else
                    for (previous = 0; previous < 256; ++previous) {
                        uint8_t previous_mask = slot < 4
                            ? backend_matmul_num_m8(&n[j].Mat,
                                                   (uint8_t)previous)
                            : (uint8_t)previous;
                        uint8_t value = SBOX[decoded_y ^ previous_mask ^
                                             round_key_byte] ^ mask;
                        uint16_t index = (uint16_t)(((uint16_t)y << 8) |
                                                    (uint16_t)previous);
                        ctx->table2[table2_offset(t2_slot, j, index)] =
                            backend_affine_u8(&e_prime_inv[slot][j], value);
                    }
#endif
                }
            }
        } else {
            int table_slot = i - 4;
            M32 ql;
            uint32_t constants[3];

            backend_matmul_mat_m32(&q[i].Mat, &linear_matrix, &ql);
            constants[0] = cus_random();
            constants[1] = cus_random();
            constants[2] = cus_random();

            for (x = 0; x < 256; ++x) {
                for (j = 0; j < 4; ++j) {
                    uint8_t round_key_byte = (uint8_t)(round_key >> (24 - 8 * j));
                    uint8_t value = backend_affine_u8(&e[i][j], (uint8_t)x);
                    uint32_t expanded;

                    value = SBOX[value ^ round_key_byte];
                    expanded = (uint32_t)value << (24 - 8 * j);
                    ctx->table3[table_slot][j][x] =
                        backend_matmul_num_m32(&ql, expanded);
                }
                for (j = 0; j < 3; ++j) {
                    ctx->table3[table_slot][j][x] ^= constants[j];
                }
                ctx->table3[table_slot][3][x] ^=
                    q[i].Vec.V ^ constants[0] ^ constants[1] ^ constants[2];
            }
        }
    }

    prepare_simd_context(ctx);
    memset(&key_schedule, 0, sizeof(key_schedule));
    ctx->magic = MASKED_WBSM4_MAGIC;
    return masked_wbsm4_validate(ctx);
}

static uint32_t lookup_masked_sbox(const masked_wbsm4_context *ctx,
                                   int round,
                                   uint32_t y,
                                   uint32_t previous_masks)
{
    uint32_t result = 0;
    int j;

    for (j = 0; j < 4; ++j) {
        uint8_t y_byte = (uint8_t)(y >> (24 - 8 * j));
        uint8_t value;

        if (round == 0 || round == 28) {
            int t1_slot = round == 0 ? 0 : 1;
            value = ctx->table1[t1_slot][j][y_byte];
        } else {
            uint8_t previous = (uint8_t)(previous_masks >> (24 - 8 * j));
            uint16_t index = (uint16_t)(((uint16_t)y_byte << 8) | previous);
            value = ctx->table2[table2_offset(table2_slot(round), j, index)];
        }
        result |= (uint32_t)value << (24 - 8 * j);
    }
    return result;
}

static uint32_t lookup_mask_value(const masked_wbsm4_context *ctx,
                                  int slot,
                                  uint32_t y)
{
    uint32_t result = 0;
    int j;

    for (j = 0; j < 4; ++j) {
        uint8_t y_byte = (uint8_t)(y >> (24 - 8 * j));
        uint8_t index = phi_power(ctx->phi[slot], y_byte, j);
        result |= (uint32_t)ctx->mask_table[slot][index] << (24 - 8 * j);
    }
    return result;
}

static uint32_t lookup_table3(const masked_wbsm4_context *ctx,
                              int round,
                              uint32_t y)
{
    int slot = round - 4;
    return ctx->table3[slot][0][(y >> 24) & 0xffu] ^
           ctx->table3[slot][1][(y >> 16) & 0xffu] ^
           ctx->table3[slot][2][(y >> 8) & 0xffu] ^
           ctx->table3[slot][3][y & 0xffu];
}

static __m512i lookup_masked_sbox16(const masked_wbsm4_context *ctx,
                                    int round,
                                    __m512i y,
                                    __m512i previous_masks)
{
    __m512i result = _mm512_setzero_si512();
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
        const __m512i y_byte = extract_byte32(y, byte_index);
        __m512i value;

        if (round == 0 || round == 28) {
            const int table_slot = round == 0 ? 0 : 1;
            value = gather_u8_16(
                ctx->table1[table_slot][byte_index], y_byte);
        } else {
            const __m512i previous = extract_byte32(
                previous_masks, byte_index);
            const __m512i index = _mm512_or_si512(
                _mm512_slli_epi32(y_byte, 8), previous);
            const uint8_t *table = ctx->table2 +
                ((size_t)table2_slot(round) * 4u + (size_t)byte_index) *
                    TABLE2_ENTRIES;
            value = gather_u8_16(table, index);
        }
        result = insert_byte32(result, value, byte_index);
    }
    return result;
}

static __m512i lookup_mask_value16(const masked_wbsm4_context *ctx,
                                   int slot,
                                   __m512i y)
{
    __m512i result = _mm512_setzero_si512();
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
        __m512i index = extract_byte32(y, byte_index);
        int power;

        for (power = 0; power < byte_index; ++power) {
            index = gather_u8_16(ctx->phi[slot], index);
        }
        result = insert_byte32(
            result, gather_u8_16(ctx->mask_table[slot], index), byte_index);
    }
    return result;
}

static __m512i lookup_table3_16(const masked_wbsm4_context *ctx,
                                int round,
                                __m512i y)
{
    const int slot = round - 4;
    __m512i result = _mm512_setzero_si512();
    int byte_index;

    for (byte_index = 0; byte_index < 4; ++byte_index) {
        result = _mm512_xor_si512(
            result,
            _mm512_i32gather_epi32(
                extract_byte32(y, byte_index),
                (const void *)ctx->table3[slot][byte_index], 4));
    }
    return result;
}

static void encrypt16(const masked_wbsm4_context *ctx,
                      const uint8_t input[16 * 16],
                      uint8_t output[16 * 16])
{
    __m512i x0 = apply_aff32_16(
        &ctx->prepared_input_encoding[0], load_word16(input, 0));
    __m512i x1 = apply_aff32_16(
        &ctx->prepared_input_encoding[1], load_word16(input, 1));
    __m512i x2 = apply_aff32_16(
        &ctx->prepared_input_encoding[2], load_word16(input, 2));
    __m512i x3 = apply_aff32_16(
        &ctx->prepared_input_encoding[3], load_word16(input, 3));
    __m512i first_masks[4] = {
        _mm512_setzero_si512(), _mm512_setzero_si512(),
        _mm512_setzero_si512(), _mm512_setzero_si512()
    };
    __m512i final_masks[4] = {
        _mm512_setzero_si512(), _mm512_setzero_si512(),
        _mm512_setzero_si512(), _mm512_setzero_si512()
    };
    __m512i mask_accumulator = _mm512_setzero_si512();
    int round;

    for (round = 0; round < 32; ++round) {
        __m512i y = apply_aff32_xor3_16(
            &ctx->prepared_part1[round][0], x1,
            &ctx->prepared_part1[round][1], x2,
            &ctx->prepared_part1[round][2], x3);
        __m512i x4;

        if (round < 4 || round >= 28) {
            const int slot = masked_slot(round);
            __m512i encoded_sbox;
            __m512i encoded_masks;
            __m512i mask_branch;

            if (round == 28) mask_accumulator = _mm512_setzero_si512();
            encoded_sbox = lookup_masked_sbox16(
                ctx, round, y, mask_accumulator);
            encoded_masks = lookup_mask_value16(ctx, slot, y);
            mask_branch = apply_aff32_16(
                &ctx->prepared_mask_branch[slot], encoded_masks);
            x4 = apply_aff32_xor2_16(
                &ctx->prepared_round_x[round], x0,
                &ctx->prepared_masked_data[slot], encoded_sbox);

            mask_accumulator = _mm512_xor_si512(
                mask_accumulator, mask_branch);
            if (round < 4) {
                first_masks[round] = mask_branch;
            } else {
                final_masks[round - 28] = mask_branch;
            }
        } else {
            __m512i table_output;

            if (round >= 4 && round <= 6) {
                __m512i masks_for_part1 = _mm512_setzero_si512();
                int mask_round;
                for (mask_round = round - 3; mask_round <= 3; ++mask_round) {
                    masks_for_part1 = _mm512_xor_si512(
                        masks_for_part1, first_masks[mask_round]);
                }
                y = _mm512_xor_si512(
                    y, apply_aff32_16(
                        &ctx->prepared_unload_part1[round - 4],
                        masks_for_part1));
            }

            table_output = lookup_table3_16(ctx, round, y);
            if (round <= 7) {
                x4 = apply_aff32_xor3_16(
                    &ctx->prepared_round_x[round], x0,
                    &ctx->prepared_round_z[round], table_output,
                    &ctx->prepared_unload_state[round - 4],
                    first_masks[round - 4]);
            } else {
                x4 = apply_aff32_xor2_16(
                    &ctx->prepared_round_x[round], x0,
                    &ctx->prepared_round_z[round], table_output);
            }
        }

        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }

    store_word16(output, 0, _mm512_xor_si512(
        apply_aff32_16(&ctx->prepared_output_decoding[0], x3),
        final_masks[3]));
    store_word16(output, 1, _mm512_xor_si512(
        apply_aff32_16(&ctx->prepared_output_decoding[1], x2),
        final_masks[2]));
    store_word16(output, 2, _mm512_xor_si512(
        apply_aff32_16(&ctx->prepared_output_decoding[2], x1),
        final_masks[1]));
    store_word16(output, 3, _mm512_xor_si512(
        apply_aff32_16(&ctx->prepared_output_decoding[3], x0),
        final_masks[0]));
}

static void encrypt_encoded(const masked_wbsm4_context *ctx,
                            const uint32_t input[4],
                            uint32_t output[4],
                            uint32_t final_masks[4])
{
    uint32_t x0 = input[0];
    uint32_t x1 = input[1];
    uint32_t x2 = input[2];
    uint32_t x3 = input[3];
    uint32_t first_masks[4] = {0, 0, 0, 0};
    uint32_t mask_accumulator = 0;
    int i;

    for (i = 0; i < 32; ++i) {
        uint32_t y = backend_affine_u32(&ctx->part1[i][0], x1) ^
                     backend_affine_u32(&ctx->part1[i][1], x2) ^
                     backend_affine_u32(&ctx->part1[i][2], x3);
        uint32_t x4;

        if (i < 4 || i >= 28) {
            int slot = masked_slot(i);
            uint32_t encoded_sbox;
            uint32_t encoded_masks;
            uint32_t data_branch;
            uint32_t mask_branch;

            if (i == 28) {
                mask_accumulator = 0;
            }
            encoded_sbox = lookup_masked_sbox(ctx, i, y, mask_accumulator);
            encoded_masks = lookup_mask_value(ctx, slot, y);
            data_branch = backend_affine_u32(&ctx->masked_data[slot],
                                               encoded_sbox);
            mask_branch = backend_affine_u32(&ctx->mask_branch[slot],
                                               encoded_masks);
            x4 = backend_affine_u32(&ctx->round_x[i], x0) ^ data_branch;

            mask_accumulator ^= mask_branch;
            if (i < 4) {
                first_masks[i] = mask_branch;
            } else {
                final_masks[i - 28] = mask_branch;
            }
        } else {
            uint32_t table_output;

            if (i >= 4 && i <= 6) {
                uint32_t masks_for_part1 = 0;
                int mask_round;
                for (mask_round = i - 3; mask_round <= 3; ++mask_round) {
                    masks_for_part1 ^= first_masks[mask_round];
                }
                y ^= backend_matmul_num_m32(&ctx->unload_part1[i - 4],
                                            masks_for_part1);
            }

            table_output = lookup_table3(ctx, i, y);
            if (i <= 7) {
                x4 = backend_affine_u32(&ctx->round_x[i], x0) ^
                     backend_affine_u32(&ctx->unload_state[i - 4],
                                        first_masks[i - 4]) ^
                     backend_affine_u32(&ctx->round_z[i], table_output);
            } else {
                x4 = backend_affine_u32(&ctx->round_x[i], x0) ^
                     backend_affine_u32(&ctx->round_z[i], table_output);
            }
        }

        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }

    output[0] = x3;
    output[1] = x2;
    output[2] = x1;
    output[3] = x0;
}

int masked_wbsm4_encrypt(const masked_wbsm4_context *ctx,
                         const uint8_t input[16],
                         uint8_t output[16])
{
    uint32_t encoded_input[4];
    uint32_t encoded_output[4];
    uint32_t final_masks[4] = {0, 0, 0, 0};
    int i;

    if (ctx == NULL || input == NULL || output == NULL) {
        return MASKED_WBSM4_BAD_ARGUMENT;
    }
    if (ctx->magic != MASKED_WBSM4_MAGIC) {
        return MASKED_WBSM4_NOT_GENERATED;
    }

    for (i = 0; i < 4; ++i) {
        encoded_input[i] = backend_affine_u32(&ctx->input_encoding[i],
                                              load_be32(input + 4 * i));
    }
    encrypt_encoded(ctx, encoded_input, encoded_output, final_masks);

    for (i = 0; i < 4; ++i) {
        uint32_t decoded = backend_affine_u32(&ctx->output_decoding[i],
                                              encoded_output[i]);
        decoded ^= final_masks[3 - i];
        store_be32(decoded, output + 4 * i);
    }
    return MASKED_WBSM4_OK;
}

int masked_wbsm4_encrypt_many_scalar(const masked_wbsm4_context *ctx,
                                     const uint8_t *input,
                                     uint8_t *output,
                                     size_t block_count)
{
    size_t block;

    if (ctx == NULL || input == NULL || output == NULL) {
        return MASKED_WBSM4_BAD_ARGUMENT;
    }
    for (block = 0; block < block_count; ++block) {
        const int status = masked_wbsm4_encrypt(
            ctx, input + block * 16, output + block * 16);
        if (status != MASKED_WBSM4_OK) return status;
    }
    return MASKED_WBSM4_OK;
}

int masked_wbsm4_encrypt_many(const masked_wbsm4_context *ctx,
                              const uint8_t *input,
                              uint8_t *output,
                              size_t block_count)
{
    if (ctx == NULL || input == NULL || output == NULL) {
        return MASKED_WBSM4_BAD_ARGUMENT;
    }
    if (ctx->magic != MASKED_WBSM4_MAGIC) {
        return MASKED_WBSM4_NOT_GENERATED;
    }
    while (block_count >= 16) {
        encrypt16(ctx, input, output);
        input += 16 * 16;
        output += 16 * 16;
        block_count -= 16;
    }
    return masked_wbsm4_encrypt_many_scalar(
        ctx, input, output, block_count);
}

int masked_wbsm4_validate(const masked_wbsm4_context *ctx)
{
    int i;

    if (ctx == NULL || ctx->table2 == NULL) {
        return MASKED_WBSM4_BAD_ARGUMENT;
    }
    for (i = 0; i < MASKED_ROUNDS; ++i) {
        if (!permutation_is_valid(ctx->phi[i])) {
            return MASKED_WBSM4_INVALID_TABLES;
        }
    }
    return MASKED_WBSM4_OK;
}

size_t masked_wbsm4_lookup_table_bytes(void)
{
    return sizeof(((masked_wbsm4_context *)0)->table1) + TABLE2_BYTES +
           sizeof(((masked_wbsm4_context *)0)->table3) +
           sizeof(((masked_wbsm4_context *)0)->mask_table) +
           sizeof(((masked_wbsm4_context *)0)->phi);
}

size_t masked_wbsm4_allocated_bytes(const masked_wbsm4_context *ctx)
{
    return ctx == NULL ? 0 : sizeof(*ctx) + TABLE2_BYTES;
}

int masked_wbsm4_generate(masked_wbsm4_context *ctx,
                          const uint8_t key[16],
                          uint32_t random_seed)
{
    return masked_wbsm4_generate_direction(ctx, key, random_seed, 0);
}
