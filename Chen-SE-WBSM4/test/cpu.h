#ifndef RELEASE_CPU_H
#define RELEASE_CPU_H
#include <stdio.h>
#include <cpuid.h>
static int release_cpu_supported(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned a,b,c,d;
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) return 0;
#ifdef RELEASE_REQUIRE_WIDE
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl")) return 0;
    __cpuid_count(7,0,a,b,c,d);
    if (!(c&(1u<<8)) || !(c&(1u<<14)) || !(c&(1u<<1)) || !(c&(1u<<12))) return 0;
#endif
    return 1;
#else
    return 0;
#endif
}
#endif
