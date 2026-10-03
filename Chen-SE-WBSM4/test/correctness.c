#include "bench_adapter.h"
#include "release_adapter.h"
#include "cpu.h"
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
#ifndef BENCH_DECRYPT
#define BENCH_DECRYPT 0
#endif
static int oracle(const uint8_t *key,const uint8_t *in,uint8_t *out,size_t n){
    EVP_CIPHER_CTX*c=EVP_CIPHER_CTX_new();int l=0,t=0;
    int ok=c && EVP_CipherInit_ex(c,EVP_sm4_ecb(),NULL,key,NULL,!BENCH_DECRYPT)==1 &&
       EVP_CIPHER_CTX_set_padding(c,0)==1 && EVP_CipherUpdate(c,out,&l,in,n)==1 &&
       EVP_CipherFinal_ex(c,out+l,&t)==1;
    EVP_CIPHER_CTX_free(c);return ok;
}
int main(void){
    if(!release_cpu_supported()){puts("SKIP: unsupported CPU");return 77;}
    if(bench_adapter_init() || bench_adapter_selftest())return 1;
    uint8_t key[16]={1,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10};
    const size_t sizes[]={1,7,8,9,15,16,17,31,32,33,65};
    uint8_t in[1040],out[1040],want[1040];
    for(unsigned instance=0;instance<3;instance++){
        release_cleanup();
        for(int j=0;j<16;j++)key[j]^=(uint8_t)(instance*17+j);
        if(release_generate(key,12345+instance,BENCH_DECRYPT))return 2;
#ifdef RELEASE_DYNAMIC
        if(release_extra_validation())return 3;
#else
        for(size_t z=0;z<sizeof(sizes)/sizeof(sizes[0]);z++){
            size_t n=sizes[z]*16;
            for(size_t j=0;j<n;j++)in[j]=(uint8_t)(j*13+instance*29+(j>>8));
            if(!oracle(key,in,want,n))return 4;
            release_execute(in,out,sizes[z]);
            if(memcmp(out,want,n)){fprintf(stderr,"Mismatch: instance=%u blocks=%zu\n",instance,sizes[z]);return 5;}
        }
#endif
    }
    release_cleanup();printf("PASS: %s %s direction=%s, 3 keys, batch boundaries and independent reference\n",bench_adapter_algorithm(),bench_adapter_variant(),BENCH_DECRYPT?"decrypt":"encrypt");return 0;
}
