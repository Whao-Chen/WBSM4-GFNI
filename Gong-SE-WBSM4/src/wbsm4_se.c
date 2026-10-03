#include "wbsm4_se.h"

static void dump_hex(uint8_t * h, int len)
{
    while(len--)
    {   
        printf("%02hhx ",*h++);
        if(len%32==0) printf("\n");
    }
    printf("\n");
}

static Aff8 A[2039], B[2039];

wbsm4_se_context *wbsm4_se_context_init(int encmode) {
    wbsm4_se_context *ctx = malloc(sizeof(wbsm4_se_context));
    if (ctx == NULL) {
        goto cleanup;
    }
    memset(ctx, 0, sizeof(wbsm4_se_context));
    ctx->encmode = encmode;
    return ctx;
cleanup:
    return NULL;
}

void wbsm4_se_context_free(wbsm4_se_context *ctx) {
    memset(ctx, 0, sizeof(wbsm4_se_context));
}

int wbsm4_se_gen_table(wbsm4_se_context *ctx, const uint8_t *key) {
    int i;
    Aff32 P[36];
    Aff32 K[32];
    Aff32 L;
    Aff32 P_inv[36];
    Aff32 AA[32], BB[32];

    sm4_context *sm4_key = sm4_context_init();
    if (ctx->encmode) {
        sm4_setkey_enc(sm4_key, key);
    } else {
        sm4_setkey_dec(sm4_key, key);
    }
    //! test
    InitRandom(1);
    // InitRandom(((unsigned int) time(NULL)));

    SE_initial(A,B);

    //affine L for linear layer
    L.Mat = L_matrix;
    L.Vec.V = 0;

    // !test
    for (i = 0; i < 36; i++) {
        //affine P
        genaffinepairM32(&P[i], &P_inv[i]);
    }

    for (i = 0; i < 32; i++) {
        // combine 4 A8 to 1 A32
        int a0, a1, a2, a3;
        // !test
        a0 = cus_random() % 2039;
        a1 = cus_random() % 2039;
        a2 = cus_random() % 2039;
        a3 = cus_random() % 2039;
        affinecomM8to32(A[a0], A[a1], A[a2], A[a3], &AA[i]);
        affinecomM8to32(B[a0], B[a1], B[a2], B[a3], &BB[i]);

        //affine K for round key
        identityM32(&K[i].Mat);
        K[i].Vec.V = sm4_key->sk[i];

        //affine M
        affinemixM32(K[i], P_inv[i + 1], &ctx->M[i][0]);
        affinemixM32(AA[i], ctx->M[i][0], &ctx->M[i][0]);

        affinemixM32(K[i], P_inv[i + 2], &ctx->M[i][1]);
        affinemixM32(AA[i], ctx->M[i][1], &ctx->M[i][1]);

        affinemixM32(K[i], P_inv[i + 3], &ctx->M[i][2]);
        affinemixM32(AA[i], ctx->M[i][2], &ctx->M[i][2]);

        //affine C D, C for Xi0, D for Pi+4 L B
        affinemixM32(P[i + 4], P_inv[i], &ctx->C[i]);

        affinemixM32(L, BB[i], &ctx->D[i]);
        affinemixM32(P[i + 4], ctx->D[i], &ctx->D[i]);
        // !test
        uint32_t temp_u32 = cus_random();
        ctx->C[i].Vec.V ^= temp_u32;
        ctx->D[i].Vec.V ^= P[i + 4].Vec.V ^ temp_u32;
    }

    //external encoding
    for (i = 0; i < 4; i++) {
        ctx->SE[i].Mat = P[i].Mat;
        ctx->SE[i].Vec = P[i].Vec;

        ctx->FE[i].Mat = P_inv[35 - i].Mat;
        ctx->FE[i].Vec = P_inv[35 - i].Vec;
    }
    return 1;
}

int wbsm4_se_encrypt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx) {
    int i;
    uint32_t x0, x1, x2, x3, x4;
    uint32_t tmp;
    uint32_t xt0, xt1, xt2, xt3, xt4;
    uint32_t tmpt;

    x0 = GET32(input);
    x1 = GET32(input + 4);
    x2 = GET32(input + 8);
    x3 = GET32(input + 12);
    //加外部编码
    x0 = affineU32(ctx->SE[0], x0);
    x1 = affineU32(ctx->SE[1], x1);
    x2 = affineU32(ctx->SE[2], x2);
    x3 = affineU32(ctx->SE[3], x3);

    for (i = 0; i < 32; i++) {
        xt1 = affineU32(ctx->M[i][0], x1);
        xt2 = affineU32(ctx->M[i][1], x2);
        xt3 = affineU32(ctx->M[i][2], x3);
        x4 = xt1 ^ xt2 ^ xt3;
        x4 = (sm4_sbox[(x4 >> 24) & 0xff] << 24) | (sm4_sbox[(x4 >> 16) & 0xff] << 16) |
             (sm4_sbox[(x4 >> 8) & 0xff] << 8) | sm4_sbox[x4 & 0xff];
        xt0 = affineU32(ctx->C[i], x0);
        xt4 = affineU32(ctx->D[i], x4);
      
        // printf("in x4:%08x \n",x4);
        // printf("xt4:%08x \n",xt4);
        x4 = xt0 ^ xt4;
        // printf("x4:%08x \n\n",x4);
        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }
    //抵消最后一轮的外部编码
    x0 = affineU32(ctx->FE[3], x0);
    x1 = affineU32(ctx->FE[2], x1);
    x2 = affineU32(ctx->FE[1], x2);
    x3 = affineU32(ctx->FE[0], x3);

    PUT32(x3, output);
    PUT32(x2, output + 4);
    PUT32(x1, output + 8);
    PUT32(x0, output + 12);

    return 1;
}
// 轮函数优化
int wbsm4_se_encrypt_func_opt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx) {
    int i;
    uint32_t x0, x1, x2, x3, tmpx;
    uint32_t tmp;
    uint32_t xt0, xt1, xt2, xt3, tmpxt;
    uint32_t tmpt;

    x0 = GET32(input);
    x1 = GET32(input + 4);
    x2 = GET32(input + 8);
    x3 = GET32(input + 12);
    //加外部编码
    x0 = affineU32(ctx->SE[0], x0);
    x1 = affineU32(ctx->SE[1], x1);
    x2 = affineU32(ctx->SE[2], x2);
    x3 = affineU32(ctx->SE[3], x3);

    for (i = 0; i < 32; i+=4) {
        xt1 = affineU32(ctx->M[i][0], x1);
        xt2 = affineU32(ctx->M[i][1], x2);
        xt3 = affineU32(ctx->M[i][2], x3);
        tmpx = xt1 ^ xt2 ^ xt3;
        tmpx = (sm4_sbox[(tmpx >> 24) & 0xff] << 24) | (sm4_sbox[(tmpx >> 16) & 0xff] << 16) |
             (sm4_sbox[(tmpx >> 8) & 0xff] << 8) | sm4_sbox[tmpx & 0xff];
        xt0 = affineU32(ctx->C[i], x0);
        tmpxt = affineU32(ctx->D[i], tmpx);
      
        x0 = xt0 ^ tmpxt;

        xt0 = affineU32(ctx->M[i+1][2], x0);
        xt2 = affineU32(ctx->M[i+1][0], x2);
        xt3 = affineU32(ctx->M[i+1][1], x3);
        tmpx = xt0 ^ xt2 ^ xt3;
        tmpx = (sm4_sbox[(tmpx >> 24) & 0xff] << 24) | (sm4_sbox[(tmpx >> 16) & 0xff] << 16) |
             (sm4_sbox[(tmpx >> 8) & 0xff] << 8) | sm4_sbox[tmpx & 0xff];
        xt1 = affineU32(ctx->C[i+1], x1);
        tmpxt = affineU32(ctx->D[i+1], tmpx);
      
        x1 = xt1 ^ tmpxt;
        
        xt0 = affineU32(ctx->M[i+2][1], x0);
        xt1 = affineU32(ctx->M[i+2][2], x1);
        xt3 = affineU32(ctx->M[i+2][0], x3);
        tmpx = xt0 ^ xt1 ^ xt3;
        tmpx = (sm4_sbox[(tmpx >> 24) & 0xff] << 24) | (sm4_sbox[(tmpx >> 16) & 0xff] << 16) |
             (sm4_sbox[(tmpx >> 8) & 0xff] << 8) | sm4_sbox[tmpx & 0xff];
        xt2 = affineU32(ctx->C[i+2], x2);
        tmpxt = affineU32(ctx->D[i+2], tmpx);
      
        x2 = xt2 ^ tmpxt;

        xt0 = affineU32(ctx->M[i+3][0], x0);
        xt1 = affineU32(ctx->M[i+3][1], x1);
        xt2 = affineU32(ctx->M[i+3][2], x2);
        tmpx = xt0 ^ xt1 ^ xt2;
        tmpx = (sm4_sbox[(tmpx >> 24) & 0xff] << 24) | (sm4_sbox[(tmpx >> 16) & 0xff] << 16) |
             (sm4_sbox[(tmpx >> 8) & 0xff] << 8) | sm4_sbox[tmpx & 0xff];
        xt3 = affineU32(ctx->C[i+3], x3);
        tmpxt = affineU32(ctx->D[i+3], tmpx);
      
        x3 = xt3 ^ tmpxt;
        
    }
    //抵消最后一轮的外部编码
    x0 = affineU32(ctx->FE[3], x0);
    x1 = affineU32(ctx->FE[2], x1);
    x2 = affineU32(ctx->FE[1], x2);
    x3 = affineU32(ctx->FE[0], x3);

    PUT32(x3, output);
    PUT32(x2, output + 4);
    PUT32(x1, output + 8);
    PUT32(x0, output + 12);

    return 1;
}

int wbsm4_se_ecb_encrypt(const unsigned char *input, unsigned char *output, size_t len, wbsm4_se_context *ctx) {
    assert (len % 16 == 0);
    while (len) {
        wbsm4_se_encrypt(input,output,ctx);
        len -= 16;
        input += 16;
        output += 16;
    }
}
//int wbsm4_se_decrypt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx) {
//    return wbsm4_se_encrypt(input, output, ctx);
//}