#include "bench_adapter.h"
#include "release_adapter.h"
#include "cpu.h"
#include <openssl/evp.h>
#include <string.h>
#ifndef BENCH_DECRYPT
#define BENCH_DECRYPT 0
#endif
#ifndef BENCH_WIDTH
#define BENCH_WIDTH 1
#endif
static const uint8_t known_key[16]={1,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10};
const char *bench_adapter_algorithm(void){return BENCH_ALGORITHM;}
const char *bench_adapter_variant(void){return BENCH_VARIANT;}
const char *bench_adapter_path_kind(void){return BENCH_PATH_KIND;}
size_t bench_adapter_batch_width(void){return BENCH_WIDTH;}
int bench_adapter_init(void){
    if(!release_cpu_supported()){fprintf(stderr,"Required CPU instruction extensions are unavailable.\n");return -1;}
    return release_generate(known_key, BENCH_SEED, BENCH_DECRYPT);
}
void bench_adapter_cleanup(void){release_cleanup();}
void bench_adapter_encrypt_one(const uint8_t in[16],uint8_t out[16]){release_execute(in,out,1);}
void bench_adapter_encrypt_many(const uint8_t *in,uint8_t *out,size_t n){release_execute(in,out,n);}
static int reference(const uint8_t key[16],const uint8_t *in,uint8_t *out,size_t n,int decrypt){
    EVP_CIPHER_CTX *c=EVP_CIPHER_CTX_new();int len=0,last=0;
    int ok=c && EVP_CipherInit_ex(c,EVP_sm4_ecb(),NULL,key,NULL,!decrypt)==1 &&
       EVP_CIPHER_CTX_set_padding(c,0)==1 && EVP_CipherUpdate(c,out,&len,in,(int)n)==1 &&
       EVP_CipherFinal_ex(c,out+len,&last)==1 && (size_t)(len+last)==n;
    EVP_CIPHER_CTX_free(c);return ok;
}
int bench_adapter_selftest(void){
#ifdef RELEASE_DYNAMIC
    return release_extra_validation();
#else
    static const uint8_t ct[16]={0x68,0x1e,0xdf,0x34,0xd2,6,0x96,0x5e,0x86,0xb3,0xe9,0x4f,0x53,0x6e,0x42,0x46};
    uint8_t actual[16],expected[16];const uint8_t *in=BENCH_DECRYPT?ct:known_key;
    if(!reference(known_key,in,expected,16,BENCH_DECRYPT))return -1;
    release_execute(in,actual,1);return memcmp(actual,expected,16)?-1:0;
#endif
}
