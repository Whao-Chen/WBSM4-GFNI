#include "wbdl.h"

#include <string.h>

#include "sbox.h"
#include "sm4.h"
#include "wbdl_wbmatrixopt_backend.h"

#define GET32(p) (((uint32_t)(p)[0] << 24) | \
                  ((uint32_t)(p)[1] << 16) | \
                  ((uint32_t)(p)[2] << 8) | \
                  ((uint32_t)(p)[3]))

#define PUT32(v, p) do {             \
    (p)[0] = (uint8_t)((v) >> 24);   \
    (p)[1] = (uint8_t)((v) >> 16);   \
    (p)[2] = (uint8_t)((v) >> 8);    \
    (p)[3] = (uint8_t)(v);           \
} while (0)

static const M32 SM4_L_MATRIX = {
    .M = {
        0xA0202080, 0x50101040, 0x28080820, 0x14040410,
        0x0A020208, 0x05010104, 0x02808082, 0x01404041,
        0x80A02020, 0x40501010, 0x20280808, 0x10140404,
        0x080A0202, 0x04050101, 0x82028080, 0x41014040,
        0x2080A020, 0x10405010, 0x08202808, 0x04101404,
        0x02080A02, 0x01040501, 0x80820280, 0x40410140,
        0x202080A0, 0x10104050, 0x08082028, 0x04041014,
        0x0202080A, 0x01010405, 0x80808202, 0x40404101
    }
};

typedef struct {
    uint32_t value;
} local_rng;

static uint32_t rng_next(local_rng *rng)
{
    uint32_t x = rng->value;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->value = x;
    return x;
}

static uint32_t rotl32(uint32_t x, unsigned int n)
{
    return (x << n) | (x >> (32U - n));
}

static uint32_t sm4_l(uint32_t x)
{
    return x ^ rotl32(x, 2) ^ rotl32(x, 10) ^
           rotl32(x, 18) ^ rotl32(x, 24);
}

static uint32_t sm4_t(uint32_t x)
{
    uint32_t s = ((uint32_t)SBOX[(x >> 24) & 0xff] << 24) |
                 ((uint32_t)SBOX[(x >> 16) & 0xff] << 16) |
                 ((uint32_t)SBOX[(x >> 8) & 0xff] << 8) |
                 (uint32_t)SBOX[x & 0xff];

    return sm4_l(s);
}

static uint8_t select_table(uint32_t x2, uint32_t x3, uint32_t factor)
{
    return (uint8_t)(((x2 ^ x3 ^ factor) >> 24) % WBDL_TABLE_COUNT);
}

static uint32_t lookup_t(const wbdl_context *ctx, uint8_t table_id,
                         uint32_t encoded_input)
{
    return ctx->table[table_id][0][(encoded_input >> 24) & 0xff] ^
           ctx->table[table_id][1][(encoded_input >> 16) & 0xff] ^
           ctx->table[table_id][2][(encoded_input >> 8) & 0xff] ^
           ctx->table[table_id][3][encoded_input & 0xff];
}

static void invert_aff32(const Aff32 *affine, Aff32 *inverse)
{
    invsM32(affine->Mat, &inverse->Mat);
    inverse->Vec.V = wbdl_opt_mat_mul_num32(&inverse->Mat, affine->Vec.V);
}

static void select_key_material(wbdl_context *ctx,
                                const uint8_t master_keys[WBDL_MASTER_KEY_COUNT][16],
                                local_rng *rng)
{
    sm4_context key_schedule[WBDL_MASTER_KEY_COUNT];
    uint32_t pool[WBDL_MASTER_KEY_COUNT * WBDL_ROUNDS];
    uint8_t order[WBDL_MASTER_KEY_COUNT * WBDL_ROUNDS];
    int i;
    int j;

    for (i = 0; i < WBDL_MASTER_KEY_COUNT; ++i) {
        sm4_setkey_enc(&key_schedule[i], (uint8_t *)master_keys[i]);
        for (j = 0; j < WBDL_ROUNDS; ++j) {
            pool[i * WBDL_ROUNDS + j] = key_schedule[i].sk[j];
            order[i * WBDL_ROUNDS + j] = (uint8_t)(i * WBDL_ROUNDS + j);
        }
    }

    for (i = (int)(sizeof(order) / sizeof(order[0])) - 1; i > 0; --i) {
        int swap_index = (int)(rng_next(rng) % (uint32_t)(i + 1));
        uint8_t tmp = order[i];
        order[i] = order[swap_index];
        order[swap_index] = tmp;
    }

    for (i = 0; i < WBDL_TABLE_COUNT; ++i) {
        ctx->table_key[i] = pool[order[i]];
    }
    for (i = 0; i < WBDL_ROUNDS; ++i) {
        ctx->selection_factor[i] = pool[order[WBDL_TABLE_COUNT + i]];
    }
}

static void generate_table(wbdl_context *ctx, int table_id,
                           const Aff8 input_encoding[4], Aff32 output_encoding,
                           local_rng *rng)
{
    M32 encoded_l;
    uint32_t shares[3];
    uint32_t output_xor[4];

    wbdl_opt_mat_mul_mat32(&output_encoding.Mat, &SM4_L_MATRIX, &encoded_l);
    shares[0] = rng_next(rng);
    shares[1] = rng_next(rng);
    shares[2] = rng_next(rng);
    output_xor[0] = shares[0];
    output_xor[1] = shares[1];
    output_xor[2] = shares[2];
    output_xor[3] = output_encoding.Vec.V ^
                    shares[0] ^ shares[1] ^ shares[2];

    wbdl_gfni_generate_table(ctx->table[table_id], input_encoding,
                             ctx->table_key[table_id], &encoded_l,
                             output_xor);
}

static void prepare_gfni_context(wbdl_context *ctx)
{
    int h;
    int i;

    for (h = 0; h < WBDL_TABLE_COUNT; ++h) {
        for (i = 0; i < WBDL_INPUT_STATE_COUNT; ++i) {
            wbdl_gfni_prepare_aff32(&ctx->input_mix[h][i],
                                    &ctx->gfni.input_mix[h][i]);
        }
    }
    for (i = 0; i < WBDL_ROUNDS; ++i) {
        wbdl_gfni_prepare_aff32(&ctx->round_x0_mix[i],
                                &ctx->gfni.round_x0_mix[i]);
        wbdl_gfni_prepare_aff32(&ctx->round_x0_mix_inv[i],
                                &ctx->gfni.round_x0_mix_inv[i]);
        for (h = 0; h < WBDL_TABLE_COUNT; ++h) {
            wbdl_gfni_prepare_aff32(&ctx->table_output_mix[i][h],
                                    &ctx->gfni.table_output_mix[i][h]);
        }
    }
    for (i = 0; i < WBDL_STATE_COUNT; ++i) {
        wbdl_gfni_prepare_aff32(&ctx->state_encoding[i],
                                &ctx->gfni.state_encoding[i]);
        wbdl_gfni_prepare_aff32(&ctx->state_decoding[i],
                                &ctx->gfni.state_decoding[i]);
    }
}

size_t wbdl_context_size(void)
{
    return sizeof(wbdl_context);
}

int wbdl_generate(wbdl_context *ctx,
                  const uint8_t master_keys[WBDL_MASTER_KEY_COUNT][16],
                  uint32_t seed)
{
    Aff8 table_input[WBDL_TABLE_COUNT][4];
    Aff8 table_input_inv[WBDL_TABLE_COUNT][4];
    Aff32 table_input_inv32[WBDL_TABLE_COUNT];
    Aff32 table_output[WBDL_TABLE_COUNT];
    Aff32 table_output_inv[WBDL_TABLE_COUNT];
    local_rng rng;
    int h;
    int i;
    int j;

    if (ctx == NULL || master_keys == NULL) {
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    rng.value = seed != 0 ? seed : 0x6d2b79f5U;
    select_key_material(ctx, master_keys, &rng);

    SetRandSeed(seed ^ 0x5742444cU);
    InitRandom(seed ^ 0x9e3779b9U);

    for (i = 0; i < WBDL_STATE_COUNT; ++i) {
        wbdl_opt_gen_affine32(&ctx->state_encoding[i],
                              &ctx->state_decoding[i]);
    }

    for (h = 0; h < WBDL_TABLE_COUNT; ++h) {
        for (j = 0; j < 4; ++j) {
            wbdl_opt_gen_affine8(&table_input[h][j],
                                 &table_input_inv[h][j]);
        }
        affinecomM8to32(table_input_inv[h][0], table_input_inv[h][1],
                        table_input_inv[h][2], table_input_inv[h][3],
                        &table_input_inv32[h]);

        wbdl_opt_gen_affine32(&table_output[h], &table_output_inv[h]);
        generate_table(ctx, h, table_input[h], table_output[h], &rng);

        for (i = 1; i <= WBDL_INPUT_STATE_COUNT; ++i) {
            wbdl_opt_affine_mix32(&table_input_inv32[h],
                                  &ctx->state_decoding[i],
                                  &ctx->input_mix[h][i - 1]);
        }
    }

    for (i = 0; i < WBDL_ROUNDS; ++i) {
        uint32_t split_constant = rng_next(&rng);

        wbdl_opt_affine_mix32(&ctx->state_encoding[i + 4],
                              &ctx->state_decoding[i],
                              &ctx->round_x0_mix[i]);
        ctx->round_x0_mix[i].Vec.V ^= split_constant;

        for (h = 0; h < WBDL_TABLE_COUNT; ++h) {
            wbdl_opt_affine_mix32(&ctx->state_encoding[i + 4],
                                  &table_output_inv[h],
                                  &ctx->table_output_mix[i][h]);
            ctx->table_output_mix[i][h].Vec.V ^=
                ctx->state_encoding[i + 4].Vec.V ^ split_constant;
        }

        invert_aff32(&ctx->round_x0_mix[i], &ctx->round_x0_mix_inv[i]);
    }

    prepare_gfni_context(ctx);

    return 0;
}

static void encrypt_encoded_core(const wbdl_context *ctx,
                                 const uint8_t input[16], uint8_t output[16],
                                 uint8_t trace[WBDL_ROUNDS])
{
    uint32_t x0 = GET32(input);
    uint32_t x1 = GET32(input + 4);
    uint32_t x2 = GET32(input + 8);
    uint32_t x3 = GET32(input + 12);
    int i;

    for (i = 0; i < WBDL_ROUNDS; ++i) {
        uint8_t h = select_table(x2, x3, ctx->selection_factor[i]);
        uint32_t table_input_value =
            wbdl_gfni_apply_aff32(&ctx->gfni.input_mix[h][i], x1) ^
            wbdl_gfni_apply_aff32(&ctx->gfni.input_mix[h][i + 1], x2) ^
            wbdl_gfni_apply_aff32(&ctx->gfni.input_mix[h][i + 2], x3);
        uint32_t table_output_value = lookup_t(ctx, h, table_input_value);
        uint32_t x4 = wbdl_gfni_apply_aff32(
                          &ctx->gfni.round_x0_mix[i], x0) ^
                      wbdl_gfni_apply_aff32(
                          &ctx->gfni.table_output_mix[i][h],
                          table_output_value);

        if (trace != NULL) {
            trace[i] = h;
        }

        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }

    PUT32(x3, output);
    PUT32(x2, output + 4);
    PUT32(x1, output + 8);
    PUT32(x0, output + 12);
}

void wbdl_encrypt_encoded(const wbdl_context *ctx,
                           const uint8_t input[16], uint8_t output[16])
{
    encrypt_encoded_core(ctx, input, output, NULL);
}

void wbdl_decrypt_encoded(const wbdl_context *ctx,
                           const uint8_t input[16], uint8_t output[16])
{
    uint32_t x_ip4 = GET32(input);
    uint32_t x_ip3 = GET32(input + 4);
    uint32_t x_ip2 = GET32(input + 8);
    uint32_t x_ip1 = GET32(input + 12);
    int i;

    for (i = WBDL_ROUNDS - 1; i >= 0; --i) {
        uint8_t h = select_table(x_ip2, x_ip3, ctx->selection_factor[i]);
        uint32_t table_input_value =
            wbdl_gfni_apply_aff32(&ctx->gfni.input_mix[h][i], x_ip1) ^
            wbdl_gfni_apply_aff32(&ctx->gfni.input_mix[h][i + 1], x_ip2) ^
            wbdl_gfni_apply_aff32(&ctx->gfni.input_mix[h][i + 2], x_ip3);
        uint32_t table_output_value = lookup_t(ctx, h, table_input_value);
        uint32_t encoded_t = wbdl_gfni_apply_aff32(
            &ctx->gfni.table_output_mix[i][h], table_output_value);
        uint32_t x_i = wbdl_gfni_apply_aff32(
            &ctx->gfni.round_x0_mix_inv[i], x_ip4 ^ encoded_t);

        x_ip4 = x_ip3;
        x_ip3 = x_ip2;
        x_ip2 = x_ip1;
        x_ip1 = x_i;
    }

    PUT32(x_ip1, output);
    PUT32(x_ip2, output + 4);
    PUT32(x_ip3, output + 8);
    PUT32(x_ip4, output + 12);
}

void wbdl_encrypt_block(const wbdl_context *ctx,
                         const uint8_t input[16], uint8_t output[16])
{
    uint8_t encoded_input[16];
    uint8_t encoded_output[16];
    int i;

    for (i = 0; i < 4; ++i) {
        uint32_t word = wbdl_gfni_apply_aff32(
            &ctx->gfni.state_encoding[i], GET32(input + 4 * i));
        PUT32(word, encoded_input + 4 * i);
    }

    wbdl_encrypt_encoded(ctx, encoded_input, encoded_output);

    for (i = 0; i < 4; ++i) {
        uint32_t word = wbdl_gfni_apply_aff32(
            &ctx->gfni.state_decoding[35 - i],
            GET32(encoded_output + 4 * i));
        PUT32(word, output + 4 * i);
    }
}

void wbdl_decrypt_block(const wbdl_context *ctx,
                         const uint8_t input[16], uint8_t output[16])
{
    uint8_t encoded_input[16];
    uint8_t encoded_output[16];
    int i;

    for (i = 0; i < 4; ++i) {
        uint32_t word = wbdl_gfni_apply_aff32(
            &ctx->gfni.state_encoding[35 - i], GET32(input + 4 * i));
        PUT32(word, encoded_input + 4 * i);
    }

    wbdl_decrypt_encoded(ctx, encoded_input, encoded_output);

    for (i = 0; i < 4; ++i) {
        uint32_t word = wbdl_gfni_apply_aff32(
            &ctx->gfni.state_decoding[i], GET32(encoded_output + 4 * i));
        PUT32(word, output + 4 * i);
    }
}

void wbdl_reference_encrypt(const wbdl_context *ctx,
                             const uint8_t input[16], uint8_t output[16])
{
    uint32_t x0 = GET32(input);
    uint32_t x1 = GET32(input + 4);
    uint32_t x2 = GET32(input + 8);
    uint32_t x3 = GET32(input + 12);
    int i;

    for (i = 0; i < WBDL_ROUNDS; ++i) {
        uint32_t encoded_x2 = wbdl_opt_affine_u32(
            &ctx->state_encoding[i + 2], x2);
        uint32_t encoded_x3 = wbdl_opt_affine_u32(
            &ctx->state_encoding[i + 3], x3);
        uint8_t h = select_table(encoded_x2, encoded_x3,
                                 ctx->selection_factor[i]);
        uint32_t x4 = x0 ^ sm4_t(x1 ^ x2 ^ x3 ^ ctx->table_key[h]);

        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }

    PUT32(x3, output);
    PUT32(x2, output + 4);
    PUT32(x1, output + 8);
    PUT32(x0, output + 12);
}

void wbdl_trace_tables(const wbdl_context *ctx, const uint8_t input[16],
                       uint8_t trace[WBDL_ROUNDS])
{
    uint8_t encoded_input[16];
    uint8_t ignored_output[16];
    int i;

    for (i = 0; i < 4; ++i) {
        uint32_t word = wbdl_gfni_apply_aff32(
            &ctx->gfni.state_encoding[i], GET32(input + 4 * i));
        PUT32(word, encoded_input + 4 * i);
    }
    encrypt_encoded_core(ctx, encoded_input, ignored_output, trace);
}
