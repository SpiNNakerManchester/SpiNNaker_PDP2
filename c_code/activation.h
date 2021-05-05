/*
 * Copyright (c) 2015-2021 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
