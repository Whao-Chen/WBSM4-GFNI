#include "wbsm4_adapter.h"

#include "wbsm4.h"

#include <string.h>

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

void wbsm4_adapter_generate(const uint8_t key[16])
{
    uint8_t mutable_key[16];
    memcpy(mutable_key, key, sizeof(mutable_key));
    wbsm4_gen(mutable_key);
}

void wbsm4_adapter_encrypt(const uint8_t input[16], uint8_t output[16])
{
    uint8_t encoded_input[16];
    uint8_t encoded_output[16];
    int word;

    for (word = 0; word < 4; ++word) {
        uint32_t value = load_be32(input + word * 4);
        store_be32(affineU32(SE[word], value), encoded_input + word * 4);
    }

    wbsm4_encrypt(encoded_input, encoded_output);

    for (word = 0; word < 4; ++word) {
        uint32_t value = load_be32(encoded_output + word * 4);
        store_be32(affineU32(FE[word], value), output + word * 4);
    }
}

