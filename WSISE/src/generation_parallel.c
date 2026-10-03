#include <stdint.h>
#include <time.h>

static uint32_t wsise_generation_seed = UINT32_C(0x26080511);

void wsise_set_generation_seed(uint32_t seed)
{
    wsise_generation_seed = seed;
}

#define time(ignored) ((time_t)wsise_generation_seed)
#include "reference.c"
#undef time
