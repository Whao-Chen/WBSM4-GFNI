#include "release_adapter.h"
#include "generation.h"
#include <stdlib.h>
#ifdef RELEASE_CHEN_CONSTRUCTION
int chen_gfni_prepare(void);
int chen_gfni_generate(const unsigned char key[16],unsigned int seed,int decrypt);
void chen_gfni_execute(const unsigned char*,unsigned char*,size_t);
int release_prepare(uint32_t seed,int decrypt){(void)seed;(void)decrypt;return chen_gfni_prepare();}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){return chen_gfni_generate(key,seed,decrypt);}
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){if(release_prepare(seed,decrypt))return -1;return release_construct(key,seed,decrypt);}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){chen_gfni_execute(in,out,n);}
#else
int release_prepare(uint32_t seed,int decrypt){(void)decrypt;srand(seed);return 0;}
int release_construct(const uint8_t key[16],uint32_t seed,int decrypt){(void)seed;return wbsm4_se_cl_gen_table_with_external_direction(key,0,decrypt)?0:-1;}
int release_generate(const uint8_t key[16],uint32_t seed,int decrypt){release_prepare(seed,decrypt);return release_construct(key,seed,decrypt);}
void release_execute(const uint8_t*in,uint8_t*out,size_t n){wbsm4_se_cl_ecb_encrypt(in,out,n*16);}
#endif
void release_cleanup(void){}
int release_extra_validation(void){return 0;}
