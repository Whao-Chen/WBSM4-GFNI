#ifndef WBSM4_SE_CL_H
#define WBSM4_SE_CL_H

#include <assert.h>
#include <stddef.h>
#include "generateTable.h"
#include "sm4_new.h"

#ifdef __cplusplus
extern "C" {
#endif

    // typedef struct {
        // int encmode;
        static u8 TypeI_c[16][256][16];
        static u8 TypeII_c[16][256][16];
        static u8 TypeI_a[32][16][256][16];
        static u8 TypeII_a[32][16][256][16];
        static u8 TypeI_b[31][16][256][16];
        static u8 TypeII_b[31][16][256][16];
        static u8 TypeI_d[16][256][16];
        static u8 S1_f4[32][4][256];
	    static u8 S2_f4[32][4][256];
    // }wbsm4_se_cl_context ;

    // wbsm4_se_cl_context *wbsm4_se_cl_context_init(int encmode);
  
    // void wbsm4_se_cl_context_free(wbsm4_se_cl_context *ctx);

    // int wbsm4_se_cl_gen_table(wbsm4_se_cl_context *ctx, const unsigned char *key);
    int wbsm4_se_cl_gen_table(const unsigned char *key);
    int wbsm4_se_cl_gen_table_with_external(const unsigned char *key, int external_encoding);
    unsigned long long wbsm4_se_cl_table_fingerprint(void);
    unsigned long long wbsm4_se_cl_table_bytes(void);
    size_t wbsm4_se_cl_table_region_count(void);
    const char *wbsm4_se_cl_table_region_name(size_t region);
    unsigned long long wbsm4_se_cl_table_region_fingerprint(size_t region);
    unsigned long long wbsm4_se_cl_table_region_bytes(size_t region);

    int wbsm4_se_cl_encrypt(const unsigned char *input, unsigned char *output);

    int wbsm4_se_cl_ecb_encrypt(const unsigned char *input, unsigned char *output, size_t len);
   
#ifdef __cplusplus
}
#endif

#endif //WBSM4_SE_CL_H
