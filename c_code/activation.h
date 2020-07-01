#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <stdint.h>

activation_t sigmoid       (net_t input);
net_t        inv_sigmoid   (activation_t input);

#define __SQRT_HALF     UINT32_C(3037000500)

extern uint64_t recip_normalized_root (uint32_t x);
extern uint64_t __x_u64_ulr           (uint64_t x, uint32_t y);

static inline uint64_t newton_xlr(uint32_t x, uint64_t r)
{
    register uint64_t t = __x_u64_ulr(r, x);

    t = ((uint64_t)(x) << 32) - (t >> 1);

    return t;
}

static inline int odd(int x)
{
    return (x & 1) == 1;
};

extern wchange_t sqrt_custom (lds_t x);

#endif
