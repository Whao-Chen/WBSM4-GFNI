#include "release_adapter.h"
#include "wbdl.h"
#include <stdlib.h>
#include <string.h>
static wbdl_context *context;
static int direction;
int release_prepare(uint32_t seed,int decrypt){(void)seed;direction=decrypt;context=calloc(1,wbdl_context_size());return context?0:-1;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){
    (void)decrypt;uint8_t keys[3][16];
    for(int j=0;j<3;j++)for(int i=0;i<16;i++)keys[j][i]=key[i]^(j*0x55);
    return wbdl_generate(context,keys,seed);
}
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){int s=release_prepare(seed,decrypt);return s?s:release_construct(key,seed,decrypt);}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){
    for(size_t i=0;i<n;i++){
        if(direction)wbdl_decrypt_block(context,in+16*i,out+16*i);
        else wbdl_encrypt_block(context,in+16*i,out+16*i);
    }
}
void release_cleanup(void){free(context);context=0;}
int release_extra_validation(void){
    uint8_t input[16],reference[16],cipher[16],plain[16];
    for(unsigned n=0;n<128;n++){
        for(int j=0;j<16;j++)input[j]=(uint8_t)(n*17+j*29);
        wbdl_reference_encrypt(context,input,reference);
        wbdl_encrypt_block(context,input,cipher);
        if(memcmp(reference,cipher,16))return -1;
        wbdl_decrypt_block(context,cipher,plain);
        if(memcmp(input,plain,16))return -1;
    }
    return 0;
}
