#include "masked_wbsm4_enhanced.h"

#include "WBMatrix/WBMatrix.h"
#include "sbox.h"
#include "sm4.h"

#include <stdlib.h>
#include <string.h>

#define ENHANCED_ROUNDS 32
#define TABLE2_ROUNDS 31
#define TABLE2_ENTRIES 65536u
#define TABLE2_BYTES \
    ((size_t)TABLE2_ROUNDS * 4u * (size_t)TABLE2_ENTRIES)
#define MASKED_WBSM4_ENHANCED_MAGIC UINT32_C(0x4d574245)

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
                }
            }
        }
    }

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
    return sizeof(((masked_wbsm4_enhanced_context *)0)->table1) +
           TABLE2_BYTES +
           sizeof(((masked_wbsm4_enhanced_context *)0)->mask_table) +
           sizeof(((masked_wbsm4_enhanced_context *)0)->phi);
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
