#include "release_adapter.h"
#include "generation.h"
#include "execution.h"
#include <string.h>
#include <time.h>
static Sm4Whitebox context;
static time_t fixed_time;
time_t time(time_t *p){if(p)*p=fixed_time;return fixed_time;}
int release_prepare(uint32_t seed,int decrypt){
    (void)decrypt;memset(&context,0,sizeof(context));
    fixed_time=(time_t)(0x5eed0000u^seed);SetRandSeed(seed);InitRandom(seed);return 0;
}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){
    (void)seed;sm4_wb_gen_tables_direction((uint8_t*)key,&context,decrypt);return 0;
}
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){
    release_prepare(seed,decrypt);return release_construct(key,seed,decrypt);
}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){for(size_t i=0;i<n;i++)sm4_wb_enc(in+16*i,out+16*i,&context);}
void release_cleanup(void){}
int release_extra_validation(void){return 0;}
