#ifndef WBMATRIX2_APPLICATION_BENCH_CLOCK_H
#define WBMATRIX2_APPLICATION_BENCH_CLOCK_H

#include <stdint.h>
#include <time.h>

static inline uint64_t bench_ticks_begin(void)
{
    unsigned lo, hi;
    __asm__ volatile("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t bench_ticks_end(void)
{
    unsigned lo, hi, aux;
    __asm__ volatile("rdtscp\n\tlfence"
                     : "=a"(lo), "=d"(hi), "=c"(aux) :: "memory");
    (void)aux;
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t bench_now_ns(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

#endif
