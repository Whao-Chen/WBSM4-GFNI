#include <string.h>
#define _GNU_SOURCE
#include "release_adapter.h"
#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
#include "bench_clock.h"
int main(int argc,char**argv){
    unsigned samples=21;int decrypt=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=strtoul(argv[++i],0,10);
        else if(!strcmp(argv[i],"--decrypt"))decrypt=1;
        else {fprintf(stderr,"Usage: %s [--samples N] [--decrypt]\n",argv[0]);return 2;}
    }
    if(!release_cpu_supported()){puts("SKIP: unsupported CPU");return 77;}
    uint8_t key[16]={1,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10};
    for(unsigned i=0;i<samples;i++){
        release_cleanup();
        if(release_prepare(12345+i,decrypt))return 3;
        uint64_t a=bench_ticks_begin();
        int ok=release_construct(key,12345+i,decrypt);
        uint64_t b=bench_ticks_end();
        if(ok)return 3;
        printf("CONSTRUCTION\t%u\t%llu\t%.6f\n",i,(unsigned long long)(b-a),(double)(b-a)/1000000.0);
    }
    release_cleanup();return 0;
}
