
#include "WBMatrix/WBMatrix.h"
#include "sm4.h"
#include <assert.h>
typedef unsigned char  u8;
typedef unsigned int   u32;
static Aff8 A[2039], B[2039];

#define GET32(pc)  (\
((uint32_t)(pc)[0] << 24) ^\
((uint32_t)(pc)[1] << 16) ^\
((uint32_t)(pc)[2] <<  8) ^\
((uint32_t)(pc)[3]))

#define PUT32(st, ct)\
(ct)[0] = (uint8_t)((st) >> 24);\
(ct)[1] = (uint8_t)((st) >> 16);\
(ct)[2] = (uint8_t)((st) >>  8);\
(ct)[3] = (uint8_t)(st)

typedef struct wbsm4_se_la_context
{
	uint32_t MM[32][3][4][256];
	uint32_t CC[32][4][256];
	uint32_t DD[32][4][256];
	uint32_t SEE[4][4][256];
	uint32_t FEE[4][4][256];
}wbsm4_se_la_context;

void wbsm4_se_la_gen(wbsm4_se_la_context *ctx, uint8_t *key);
void wbsm4_se_la_encrypt(unsigned char IN[], unsigned char OUT[], wbsm4_se_la_context *ctx);
int wbsm4_se_la_ecb_encrypt(wbsm4_se_la_context *ctx, const unsigned char *input, size_t ilen, unsigned char *output);
void wbsm4_se_la_free(wbsm4_se_la_context *ctx);