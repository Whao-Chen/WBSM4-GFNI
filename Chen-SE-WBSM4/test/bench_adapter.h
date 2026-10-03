#ifndef WBSM4_THROUGHPUT_BENCH_ADAPTER_H
#define WBSM4_THROUGHPUT_BENCH_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

const char *bench_adapter_algorithm(void);
const char *bench_adapter_variant(void);
const char *bench_adapter_path_kind(void);
size_t bench_adapter_batch_width(void);

int bench_adapter_init(void);
void bench_adapter_cleanup(void);
int bench_adapter_selftest(void);
void bench_adapter_encrypt_one(const uint8_t input[16], uint8_t output[16]);
void bench_adapter_encrypt_many(const uint8_t *input, uint8_t *output,
                                size_t block_count);

#endif
