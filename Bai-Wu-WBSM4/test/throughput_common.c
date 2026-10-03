#define _GNU_SOURCE

#include "bench_adapter.h"

#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

enum {
    DEFAULT_SAMPLES = 20,
    DEFAULT_MIN_TOTAL_BYTES = 1024 * 1024,
    DEFAULT_WARMUP_BYTES = 64 * 1024,
    MAX_MESSAGES = 32,
    CORRECTNESS_BLOCKS = 1024
};

static volatile uint64_t benchmark_sink;

static const size_t default_messages[] = {
    16, 64, 128, 256, 272, 512, 528, 8912, 1024 * 1024
};

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t value;
    *state += UINT64_C(0x9e3779b97f4a7c15);
    value = *state;
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static void fill_deterministic(uint8_t *buffer, size_t length, uint64_t seed)
{
    size_t offset = 0;
    while (offset < length) {
        uint64_t value = splitmix64(&seed);
        int byte;
        for (byte = 0; byte < 8 && offset < length; ++byte, ++offset) {
            buffer[offset] = (uint8_t)(value >> (8 * byte));
        }
    }
}

static uint64_t checksum(const uint8_t *buffer, size_t length)
{
    uint64_t value = UINT64_C(0x243f6a8885a308d3) ^ length;
    size_t offset;
    for (offset = 0; offset < length; ++offset) {
        value ^= (uint64_t)buffer[offset] + UINT64_C(0x9e3779b97f4a7c15) +
                 (value << 7) + (value >> 3);
    }
    return value;
}

static uint64_t rdtsc_begin(void)
{
    _mm_lfence();
    return __rdtsc();
}

static uint64_t rdtsc_end(void)
{
    unsigned int auxiliary;
    uint64_t value = __rdtscp(&auxiliary);
    _mm_lfence();
    return value;
}

static double elapsed_seconds(const struct timespec *begin,
                              const struct timespec *end)
{
    return (double)(end->tv_sec - begin->tv_sec) +
           (double)(end->tv_nsec - begin->tv_nsec) / 1000000000.0;
}

static int parse_size(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        return -1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int parse_messages(char *text, size_t messages[MAX_MESSAGES],
                          size_t *message_count)
{
    char *save = NULL;
    char *token = strtok_r(text, ",", &save);
    size_t count = 0;
    while (token != NULL) {
        size_t value;
        if (count == MAX_MESSAGES || parse_size(token, &value) != 0 ||
            value % 16 != 0) {
            return -1;
        }
        messages[count++] = value;
        token = strtok_r(NULL, ",", &save);
    }
    if (count == 0) {
        return -1;
    }
    *message_count = count;
    return 0;
}

static int common_correctness_gate(void)
{
    const size_t bytes = CORRECTNESS_BLOCKS * 16u;
    uint8_t *input = (uint8_t *)malloc(bytes);
    uint8_t *scalar = (uint8_t *)malloc(bytes);
    uint8_t *selected = (uint8_t *)malloc(bytes);
    size_t block;
    int result = 0;

    if (input == NULL || scalar == NULL || selected == NULL) {
        free(input);
        free(scalar);
        free(selected);
        return -1;
    }
    fill_deterministic(input, bytes, UINT64_C(0x260824534d340001));
    for (block = 0; block < CORRECTNESS_BLOCKS; ++block) {
        bench_adapter_encrypt_one(input + block * 16,
                                  scalar + block * 16);
    }
    bench_adapter_encrypt_many(input, selected, CORRECTNESS_BLOCKS);
    if (memcmp(scalar, selected, bytes) != 0) {
        result = -1;
    }
    benchmark_sink ^= checksum(scalar, bytes) ^ checksum(selected, bytes);
    free(input);
    free(scalar);
    free(selected);
    return result;
}

static void warm_up(const uint8_t *input, uint8_t *output,
                    size_t message_bytes, size_t warmup_bytes)
{
    size_t calls = (warmup_bytes + message_bytes - 1) / message_bytes;
    size_t call;
    for (call = 0; call < calls; ++call) {
        bench_adapter_encrypt_many(input, output, message_bytes / 16);
    }
    benchmark_sink ^= checksum(output, message_bytes);
}

static int run_message(size_t message_bytes, size_t samples,
                       size_t min_total_bytes, size_t warmup_bytes)
{
    size_t calls = (min_total_bytes + message_bytes - 1) / message_bytes;
    size_t processed_bytes = calls * message_bytes;
    size_t processed_blocks = processed_bytes / 16;
    uint8_t *input = NULL;
    uint8_t *output = NULL;
    size_t sample;

    if (posix_memalign((void **)&input, 64, processed_bytes) != 0 ||
        posix_memalign((void **)&output, 64, processed_bytes) != 0) {
        free(input);
        free(output);
        return -1;
    }

    fill_deterministic(input, processed_bytes,
                       UINT64_C(0x2608240000000000) ^ message_bytes);
    memset(output, 0, processed_bytes);
    warm_up(input, output, message_bytes, warmup_bytes);

    for (sample = 0; sample < samples; ++sample) {
        struct timespec begin;
        struct timespec end;
        uint64_t begin_cycles;
        uint64_t end_cycles;
        uint64_t digest;
        double seconds;
        double mbps;
        size_t call;

        fill_deterministic(input, processed_bytes,
                           UINT64_C(0x2608241000000000) ^
                           ((uint64_t)message_bytes << 17) ^ sample);
        clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
        begin_cycles = rdtsc_begin();
        for (call = 0; call < calls; ++call) {
            const size_t offset = call * message_bytes;
            bench_adapter_encrypt_many(input + offset, output + offset,
                                       message_bytes / 16);
        }
        end_cycles = rdtsc_end();
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        seconds = elapsed_seconds(&begin, &end);
        mbps = ((double)processed_bytes * 8.0 / 1000000.0) / seconds;
        digest = checksum(output, processed_bytes);
        benchmark_sink ^= digest + sample + message_bytes;
        printf("ROW\t%s\t%s\t%s\t%zu\t%zu\t%zu\t%zu\t%zu\t%.9f\t%.6f\t%" PRIu64 "\t%.6f\t%016" PRIx64 "\n",
               bench_adapter_algorithm(), bench_adapter_variant(),
               bench_adapter_path_kind(), bench_adapter_batch_width(),
               message_bytes, sample, processed_bytes, calls, seconds, mbps,
               end_cycles - begin_cycles,
               (double)(end_cycles - begin_cycles) / processed_blocks,
               digest);
        fflush(stdout);
    }

    free(input);
    free(output);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [--correctness-only] [--samples N] "
            "[--min-total-bytes N] [--warmup-bytes N] "
            "[--messages comma-separated-bytes]\n",
            program);
}

int main(int argc, char **argv)
{
    size_t samples = DEFAULT_SAMPLES;
    size_t min_total_bytes = DEFAULT_MIN_TOTAL_BYTES;
    size_t warmup_bytes = DEFAULT_WARMUP_BYTES;
    size_t messages[MAX_MESSAGES];
    size_t message_count = sizeof(default_messages) / sizeof(default_messages[0]);
    int correctness_only = 0;
    int argument;
    int result = 0;

    memcpy(messages, default_messages, sizeof(default_messages));
    for (argument = 1; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--correctness-only") == 0) {
            correctness_only = 1;
        } else if (argument + 1 < argc &&
                   strcmp(argv[argument], "--samples") == 0) {
            if (parse_size(argv[++argument], &samples) != 0) result = -1;
        } else if (argument + 1 < argc &&
                   strcmp(argv[argument], "--min-total-bytes") == 0) {
            if (parse_size(argv[++argument], &min_total_bytes) != 0) result = -1;
        } else if (argument + 1 < argc &&
                   strcmp(argv[argument], "--warmup-bytes") == 0) {
            if (parse_size(argv[++argument], &warmup_bytes) != 0) result = -1;
        } else if (argument + 1 < argc &&
                   strcmp(argv[argument], "--messages") == 0) {
            if (parse_messages(argv[++argument], messages, &message_count) != 0)
                result = -1;
        } else {
            result = -1;
        }
    }
    if (result != 0) {
        usage(argv[0]);
        return 2;
    }

    if (bench_adapter_init() != 0) {
        fprintf(stderr, "adapter initialization failed\n");
        return 2;
    }
    if (bench_adapter_selftest() != 0 || common_correctness_gate() != 0) {
        fprintf(stderr, "correctness gate failed for %s/%s\n",
                bench_adapter_algorithm(), bench_adapter_variant());
        bench_adapter_cleanup();
        return 3;
    }

    printf("CORRECTNESS\t%s\t%s\tPASS\tblocks=%d\n",
           bench_adapter_algorithm(), bench_adapter_variant(),
           CORRECTNESS_BLOCKS);
    printf("META\talgorithm=%s\tvariant=%s\tpath_kind=%s\tbatch_width=%zu\tcpu=%d\tsamples=%zu\tmin_total_bytes=%zu\n",
           bench_adapter_algorithm(), bench_adapter_variant(),
           bench_adapter_path_kind(), bench_adapter_batch_width(), sched_getcpu(),
           samples, min_total_bytes);
    printf("SCHEMA\trecord\talgorithm\tvariant\tpath_kind\tbatch_width\tmessage_bytes\tsample\tprocessed_bytes\tcalls\tseconds\tmbps\ttsc_cycles\tcycles_per_block\tchecksum\n");
    fflush(stdout);

    if (!correctness_only) {
        size_t index;
        for (index = 0; index < message_count; ++index) {
            if (run_message(messages[index], samples, min_total_bytes,
                            warmup_bytes) != 0) {
                result = 2;
                break;
            }
        }
    }

    printf("SINK\t%" PRIu64 "\n", benchmark_sink);
    bench_adapter_cleanup();
    return result;
}
