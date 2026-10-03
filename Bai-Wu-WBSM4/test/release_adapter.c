#include "release_adapter.h"
#include "generation.h"
#include "wbsm4_adapter.h"
#ifdef RELEASE_HAVE_MANY
#include "baiwu_gfni_simd.h"
extern void baiwu_set_generation_seed(uint32_t);
#endif
#include <string.h>
int release_prepare(uint32_t seed,int decrypt){(void)seed;(void)decrypt;return 0;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){
    uint8_t mutable_key[16];memcpy(mutable_key,key,16);
#ifdef RELEASE_HAVE_MANY
    baiwu_set_generation_seed(seed);
#endif
    wbsm4_gen_direction(mutable_key,decrypt);
#ifdef RELEASE_HAVE_MANY
    baiwu_gfni_prepare();
#endif
    return 0;
}
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){release_prepare(seed,decrypt);return release_construct(key,seed,decrypt);}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){
#ifdef RELEASE_HAVE_MANY
    wbsm4_adapter_encrypt_many(in,out,n);
#else
    for(size_t i=0;i<n;i++)wbsm4_adapter_encrypt(in+16*i,out+16*i);
#endif
}
void release_cleanup(void){}
int release_extra_validation(void){return 0;}
