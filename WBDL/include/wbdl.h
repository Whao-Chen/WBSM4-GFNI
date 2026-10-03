#ifndef WBDL_H
#define WBDL_H

#include <stddef.h>
#include <stdint.h>

#include "WBMatrix/WBMatrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WBDL_ROUNDS 32
#define WBDL_TABLE_COUNT 37
#define WBDL_MASTER_KEY_COUNT 3
#define WBDL_STATE_COUNT 36
#define WBDL_INPUT_STATE_COUNT 34

typedef struct {
    Aff32 input_mix[WBDL_TABLE_COUNT][WBDL_INPUT_STATE_COUNT];
    Aff32 round_x0_mix[WBDL_ROUNDS];
    Aff32 round_x0_mix_inv[WBDL_ROUNDS];
    Aff32 table_output_mix[WBDL_ROUNDS][WBDL_TABLE_COUNT];
    uint32_t table[WBDL_TABLE_COUNT][4][256];

    Aff32 state_encoding[WBDL_STATE_COUNT];
    Aff32 state_decoding[WBDL_STATE_COUNT];
    uint32_t table_key[WBDL_TABLE_COUNT];
    uint32_t selection_factor[WBDL_ROUNDS];
} wbdl_context;

size_t wbdl_context_size(void);

int wbdl_generate(wbdl_context *ctx,
                  const uint8_t master_keys[WBDL_MASTER_KEY_COUNT][16],
                  uint32_t seed);

void wbdl_encrypt_encoded(const wbdl_context *ctx,
                           const uint8_t input[16],
                           uint8_t output[16]);

void wbdl_decrypt_encoded(const wbdl_context *ctx,
                           const uint8_t input[16],
                           uint8_t output[16]);

void wbdl_encrypt_block(const wbdl_context *ctx,
                         const uint8_t input[16],
                         uint8_t output[16]);

void wbdl_decrypt_block(const wbdl_context *ctx,
                         const uint8_t input[16],
                         uint8_t output[16]);

void wbdl_reference_encrypt(const wbdl_context *ctx,
                             const uint8_t input[16],
                             uint8_t output[16]);

void wbdl_trace_tables(const wbdl_context *ctx,
                       const uint8_t input[16],
                       uint8_t trace[WBDL_ROUNDS]);

#ifdef __cplusplus
}
#endif

#endif
