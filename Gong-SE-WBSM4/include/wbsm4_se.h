#ifndef WBSM4_SE_H
#define WBSM4_SE_H

#include <WBMatrix/WBMatrix.h>
#include <string.h>
#include <assert.h>
#include <sm4.h>
#include <wbsm4_se_local.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        int encmode;
        Aff32 M[32][3];
        Aff32 C[32];
        Aff32 D[32];
        Aff32 SE[4];
        Aff32 FE[4];
    }wbsm4_se_context ;
    /******************************************basic function**********************************************/
    /**
    * the function initializes the se_wbsm4 context
    * @param encmode encrypto mode. {WBCRYPTO_ENCRYPT_MODE: encrypto mode; WBCRYPTO_DECRYPT_MODE: decrypto mode}
    * @param dummyrounds add extra dummyrounds, 1 dummyround will be expanded to 4 rounds in the runtime
    * @return ctx Context to initialize, NULL is fault, otherwise successful
    */
    wbsm4_se_context *wbsm4_se_context_init(int encmode);

    /**
    * free se_wbsm4 context
    * @param ctx
    */
    void wbsm4_se_context_free(wbsm4_se_context *ctx);

    /**
    * the function generate key-tables by the key in the context
    * @param key key used to generate the key-tables, which need to hide
    * @param keylen key length
    * @return 1 if success, 0 if error
    */
    int wbsm4_se_gen_table(wbsm4_se_context *ctx, const uint8_t *key);

    /**
    * the function is used to encrypt(**generally not used directly**)
    * @param ctx se-wbsm4-ctx must be init
    * @param input plaintext
    * @param output ciphertext
    * @return 1 if success, 0 if error
    */
    int wbsm4_se_encrypt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx);

    /**
    * the function is used to decrypt(**generally not used directly**)
    * @param ctx se-wbsm4-ctx must be init
    * @param input ciphertext
    * @param output plaintext
    * @return 1 if success, 0 if error
    */
//    int wbsm4_se_decrypt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx);
    int wbsm4_se_ecb_encrypt(const unsigned char *input, unsigned char *output, size_t len, wbsm4_se_context *ctx);
int wbsm4_se_encrypt_func_opt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx);
   
#ifdef __cplusplus
}
#endif

#endif //WBSM4_SE_H
