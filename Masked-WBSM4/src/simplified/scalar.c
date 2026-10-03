#include "masked_wbsm4.h"

#include "WBMatrix/WBMatrix.h"
#include "matrix_bindings.h"
#include "sbox.h"
#include "sm4.h"

#include <stdlib.h>
#include <string.h>

#define MASKED_ROUNDS 8
#define TABLE1_ROUNDS 2
#define TABLE2_ROUNDS 6
#define TABLE3_ROUNDS 24
#define TABLE2_ENTRIES 65536u
#define TABLE2_BYTES ((size_t)TABLE2_ROUNDS * 4u * TABLE2_ENTRIES)
#define MASKED_WBSM4_MAGIC UINT32_C(0x4d574234)

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
    return slot < 4 ? affineU8(mask_encoding[slot], encoded) : encoded;
}

static size_t table2_offset(int slot, int byte_index, uint16_t input)
{
    return (((size_t)slot * 4u + (size_t)byte_index) * TABLE2_ENTRIES) + input;
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
        genaffinepairM32(&p[i], &p_inv[i]);
    }

    for (i = 0; i < 32; ++i) {
        for (j = 0; j < 4; ++j) {
            genaffinepairM8(&e[i][j], &e_inv[i][j]);
        }
        affinecomM8to32(e_inv[i][0], e_inv[i][1],
                        e_inv[i][2], e_inv[i][3], &e32_inv[i]);

        for (j = 0; j < 3; ++j) {
            ctx->part1[i][j] = compose_aff32(e32_inv[i], p_inv[i + j + 1]);
        }
    }

    for (j = 0; j < 4; ++j) {
        genMatpairM8(&n[j].Mat, &n_inv[j].Mat);
        n[j].Vec.V = 0;
        n_inv[j].Vec.V = 0;
    }
    affinecomM8to32(n[0], n[1], n[2], n[3], &n32);
    affinecomM8to32(n_inv[0], n_inv[1], n_inv[2], n_inv[3], &n32_inv);

    for (i = 0; i < MASKED_ROUNDS; ++i) {
        for (j = 0; j < 4; ++j) {
            genaffinepairM8(&e_prime[i][j], &e_prime_inv[i][j]);
        }
        if (i < 4) {
            genaffinepairM8(&mask_encoding[i], &mask_encoding_inv[i]);
        } else {
            identity_aff8(&mask_encoding[i]);
            identity_aff8(&mask_encoding_inv[i]);
        }
        generate_permutation(ctx->phi[i]);
        for (x = 0; x < 256; ++x) {
            uint8_t mask = (uint8_t)cus_random();
            ctx->mask_table[i][x] = i < 4
                ? affineU8(mask_encoding_inv[i], mask)
                : mask;
        }
    }

    for (i = 4; i <= 27; ++i) {
        genaffinepairM32(&q[i], &q_inv[i]);
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
                MatMulMatM32(e32_inv[i].Mat, n32.Mat, &unload_part1);
                ctx->unload_part1[i - 4] = unload_part1;
            }
            MatMulMatM32(p[i + 4].Mat, n32.Mat, &unload_state);
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
                    uint8_t value = affineU8(e[i][j], (uint8_t)x);
                    uint8_t mask = recover_mask(ctx, mask_encoding, slot, j,
                                                (uint8_t)x);
                    value = SBOX[value ^ round_key_byte] ^ mask;
                    ctx->table1[table_slot][j][x] =
                        affineU8(e_prime_inv[slot][j], value);
                }
            }
        } else if ((i >= 1 && i <= 3) || (i >= 29 && i <= 31)) {
            int slot = masked_slot(i);
            int t2_slot = table2_slot(i);
            for (j = 0; j < 4; ++j) {
                uint8_t round_key_byte = (uint8_t)(round_key >> (24 - 8 * j));
                for (x = 0; x < 256; ++x) {
                    uint8_t y = (uint8_t)x;
                    uint8_t decoded_y = affineU8(e[i][j], y);
                    uint8_t mask = recover_mask(ctx, mask_encoding, slot, j, y);
                    int previous;
                    for (previous = 0; previous < 256; ++previous) {
                        uint8_t previous_mask = slot < 4
                            ? MatMulNumM8(n[j].Mat, (uint8_t)previous)
                            : (uint8_t)previous;
                        uint8_t value = SBOX[decoded_y ^ previous_mask ^
                                             round_key_byte] ^ mask;
                        uint16_t index = (uint16_t)(((uint16_t)y << 8) |
                                                    (uint16_t)previous);
                        ctx->table2[table2_offset(t2_slot, j, index)] =
                            affineU8(e_prime_inv[slot][j], value);
                    }
                }
            }
        } else {
            int table_slot = i - 4;
            M32 ql;
            uint32_t constants[3];

            MatMulMatM32(q[i].Mat, linear_matrix, &ql);
            constants[0] = cus_random();
            constants[1] = cus_random();
            constants[2] = cus_random();

            for (x = 0; x < 256; ++x) {
                for (j = 0; j < 4; ++j) {
                    uint8_t round_key_byte = (uint8_t)(round_key >> (24 - 8 * j));
                    uint8_t value = affineU8(e[i][j], (uint8_t)x);
                    uint32_t expanded;

                    value = SBOX[value ^ round_key_byte];
                    expanded = (uint32_t)value << (24 - 8 * j);
                    ctx->table3[table_slot][j][x] =
                        MatMulNumM32(ql, expanded);
                }
                for (j = 0; j < 3; ++j) {
                    ctx->table3[table_slot][j][x] ^= constants[j];
                }
                ctx->table3[table_slot][3][x] ^=
                    q[i].Vec.V ^ constants[0] ^ constants[1] ^ constants[2];
            }
        }
    }

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
        uint32_t y = affineU32(ctx->part1[i][0], x1) ^
                     affineU32(ctx->part1[i][1], x2) ^
                     affineU32(ctx->part1[i][2], x3);
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
            data_branch = affineU32(ctx->masked_data[slot], encoded_sbox);
            mask_branch = affineU32(ctx->mask_branch[slot], encoded_masks);
            x4 = affineU32(ctx->round_x[i], x0) ^ data_branch;

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
                y ^= MatMulNumM32(ctx->unload_part1[i - 4], masks_for_part1);
            }

            table_output = lookup_table3(ctx, i, y);
            if (i <= 7) {
                x4 = affineU32(ctx->round_x[i], x0) ^
                     affineU32(ctx->unload_state[i - 4], first_masks[i - 4]) ^
                     affineU32(ctx->round_z[i], table_output);
            } else {
                x4 = affineU32(ctx->round_x[i], x0) ^
                     affineU32(ctx->round_z[i], table_output);
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
        encoded_input[i] = affineU32(ctx->input_encoding[i],
                                     load_be32(input + 4 * i));
    }
    encrypt_encoded(ctx, encoded_input, encoded_output, final_masks);

    for (i = 0; i < 4; ++i) {
        uint32_t decoded = affineU32(ctx->output_decoding[i],
                                     encoded_output[i]);
        decoded ^= final_masks[3 - i];
        store_be32(decoded, output + 4 * i);
    }
    return MASKED_WBSM4_OK;
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
