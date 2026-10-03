#include "release_adapter.h"
#include "wbsm4_se.h"
#include <stdlib.h>
#if RELEASE_GONG_IMPL == 2
#include "wbsm4_se_avx2.h"
static wbsm4_se_avx2_context *context;
int release_prepare(uint32_t seed,int decrypt){(void)seed;context=wbsm4_se_avx2_context_init(!decrypt);return context?0:-1;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){(void)seed;(void)decrypt;return wbsm4_se_avx2_gen_table(context,key)?0:-1;}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){wbsm4_se_avx2_ecb_encrypt(in,out,n*16,context);}
void release_cleanup(void){if(context){wbsm4_se_avx2_context_free(context);free(context);context=0;}}
#else
#include "gong_gfni.h"
static wbsm4_se_context *context;
static GongWbsm4SePreparedGFNI prepared;
int release_prepare(uint32_t seed,int decrypt){(void)seed;context=wbsm4_se_context_init(!decrypt);return context?0:-1;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){
    (void)seed;(void)decrypt;if(!wbsm4_se_gen_table(context,key))return -1;
#if RELEASE_GONG_IMPL == 1
    wbsm4_se_prepare_gfni_context(context,&prepared);
#endif
    return 0;
}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){
#if RELEASE_GONG_IMPL == 1
    wbsm4_se_ecb_encrypt_gfni_prepared_x8_avxpack(in,out,n*16,&prepared);
#else
    wbsm4_se_ecb_encrypt(in,out,n*16,context);
#endif
}
void release_cleanup(void){if(context){wbsm4_se_context_free(context);free(context);context=0;}}
#endif
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){int s=release_prepare(seed,decrypt);return s?s:release_construct(key,seed,decrypt);}
int release_extra_validation(void){return 0;}
