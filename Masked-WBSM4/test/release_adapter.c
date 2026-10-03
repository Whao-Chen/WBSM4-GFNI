#include "release_adapter.h"
#include "generation.h"
#ifdef RELEASE_ENHANCED
static masked_wbsm4_enhanced_context *context;
int release_prepare(uint32_t seed,int decrypt){(void)seed;(void)decrypt;context=masked_wbsm4_enhanced_create();return context?0:-1;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){return masked_wbsm4_enhanced_generate_direction(context,key,seed,decrypt);}
void release_cleanup(void){masked_wbsm4_enhanced_destroy(context);context=0;}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){
#ifdef RELEASE_HAVE_MANY
    (void)masked_wbsm4_enhanced_encrypt_many(context,in,out,n);
#else
    for(size_t i=0;i<n;i++)(void)masked_wbsm4_enhanced_encrypt(context,in+16*i,out+16*i);
#endif
}
#else
static masked_wbsm4_context *context;
int release_prepare(uint32_t seed,int decrypt){(void)seed;(void)decrypt;context=masked_wbsm4_create();return context?0:-1;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){return masked_wbsm4_generate_direction(context,key,seed,decrypt);}
void release_cleanup(void){masked_wbsm4_destroy(context);context=0;}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){
#ifdef RELEASE_HAVE_MANY
    (void)masked_wbsm4_encrypt_many(context,in,out,n);
#else
    for(size_t i=0;i<n;i++)(void)masked_wbsm4_encrypt(context,in+16*i,out+16*i);
#endif
}
#endif
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){int s=release_prepare(seed,decrypt);return s?s:release_construct(key,seed,decrypt);}
int release_extra_validation(void){return 0;}
