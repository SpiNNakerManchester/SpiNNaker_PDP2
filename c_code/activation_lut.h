/*
 * Copyright (c) 2015 The University of Manchester
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __ACTIVATION_LUT_H__
#define __ACTIVATION_LUT_H__


// -----------------------
// activation function constants
// -----------------------
// number of points in the look-up table
#define SPINN_SIGMD_RES 256

// look-up table LOGISTIC function
// not all values are used!
//TODO: should compute this from other values!
#define SPINN_SIGMD_MAX_INPUT     ( 8.0 * (1 << SPINN_NET_SHIFT))
#define SPINN_SIGMD_MIN_INPUT     (-8.0 * (1 << SPINN_NET_SHIFT))
#define SPINN_SIGMD_MAX_DERIV     (0.25 * (1 << SPINN_SHORT_ACTIV_SHIFT))
#define SPINN_SIGMD_MIN_DERIV     0
#define SPINN_SIGMD_MAX_OFFSET    (0.50 * (1 << SPINN_SHORT_ACTIV_SHIFT))
#define SPINN_SIGMD_LUT_SHIFT     (SPINN_NET_SHIFT - 5)
#define SPINN_SIGMD_LUT_IMASK     ((1 << SPINN_SIGMD_LUT_SHIFT) - 1)
#define SPINN_INVSIG_LUT_SHIFT    (SPINN_SHORT_ACTIV_SHIFT - 9)
#define SPINN_INVSIG_LUT_IMASK    ((1 << SPINN_INVSIG_LUT_SHIFT) - 1)


// -----------------------
// activation look-up tables
// -----------------------
// look-up table LOGISTIC function
const activation_t sigmoid_lut[SPINN_SIGMD_RES] =
{
  0x04000000, 0x040fffaa, 0x041ffd55, 0x042ff702,
  0x043feab3, 0x044fd66f, 0x045fb840, 0x046f8e36,
  0x047f5664, 0x048f0ee8, 0x049eb5e3, 0x04ae4983,
  0x04bdc7fc, 0x04cd2f8d, 0x04dc7e82, 0x04ebb32e,
  0x04facbf5, 0x0509c744, 0x0518a399, 0x05275f7e,
  0x0535f98a, 0x05447065, 0x0552c2c5, 0x0560ef71,
  0x056ef53d, 0x057cd311, 0x058a87e1, 0x059812b3,
  0x05a5729e, 0x05b2a6c9, 0x05bfae6a, 0x05cc88c9,
  0x05d9353d, 0x05e5b32d, 0x05f20211, 0x05fe216f,
  0x060a10de, 0x0615d002, 0x06215e8f, 0x062cbc48,
  0x0637e8fd, 0x0642e48b, 0x064daede, 0x065847ee,
  0x0662afbf, 0x066ce662, 0x0676ebf1, 0x0680c094,
  0x068a647c, 0x0693d7e5, 0x069d1b12, 0x06a62e53,
  0x06af11fe, 0x06b7c671, 0x06c04c13, 0x06c8a351,
  0x06d0cca1, 0x06d8c87c, 0x06e09763, 0x06e839dc,
  0x06efb071, 0x06f6fbb3, 0x06fe1c34, 0x0705128c,
  0x070bdf56, 0x0712832f, 0x0718feb8, 0x071f5294,
  0x07257f66, 0x072b85d7, 0x0731668c, 0x07372230,
  0x073cb96b, 0x07422ce8, 0x07477d53, 0x074cab54,
  0x0751b797, 0x0756a2c6, 0x075b6d8b, 0x0760188e,
  0x0764a477, 0x076911ee, 0x076d6197, 0x07719419,
  0x0775aa14, 0x0779a42b, 0x077d82fe, 0x07814729,
  0x0784f14a, 0x078881f9, 0x078bf9d0, 0x078f5963,
  0x0792a146, 0x0795d20b, 0x0798ec40, 0x079bf073,
  0x079edf2f, 0x07a1b8fa, 0x07a47e5b, 0x07a72fd5,
  0x07a9cde8, 0x07ac5914, 0x07aed1d5, 0x07b138a3,
  0x07b38df5, 0x07b5d242, 0x07b805fa, 0x07ba298e,
  0x07bc3d6b, 0x07be41fc, 0x07c037a9, 0x07c21eda,
  0x07c3f7f2, 0x07c5c354, 0x07c7815e, 0x07c9326d,
  0x07cad6de, 0x07cc6f09, 0x07cdfb44, 0x07cf7be4,
  0x07d0f13d, 0x07d25b9e, 0x07d3bb57, 0x07d510b3,
  0x07d65bfd, 0x07d79d7f, 0x07d8d57f, 0x07da0442,
  0x07db2a0b, 0x07dc471d, 0x07dd5bb6, 0x07de6816,
  0x07df6c78, 0x07e06917, 0x07e15e2d, 0x07e24bf2,
  0x07e3329b, 0x07e4125e, 0x07e4eb6e, 0x07e5bdfd,
  0x07e68a3c, 0x07e75059, 0x07e81082, 0x07e8cae5,
  0x07e97fad, 0x07ea2f04, 0x07ead912, 0x07eb7e01,
  0x07ec1df6, 0x07ecb917, 0x07ed4f89, 0x07ede170,
  0x07ee6eed, 0x07eef823, 0x07ef7d31, 0x07effe39,
  0x07f07b57, 0x07f0f4ab, 0x07f16a52, 0x07f1dc66,
  0x07f24b04, 0x07f2b647, 0x07f31e47, 0x07f3831f,
  0x07f3e4e5, 0x07f443b3, 0x07f49f9e, 0x07f4f8bd,
  0x07f54f26, 0x07f5a2ed, 0x07f5f427, 0x07f642e7,
  0x07f68f41, 0x07f6d947, 0x07f7210b, 0x07f7669f,
  0x07f7aa13, 0x07f7eb78, 0x07f82ade, 0x07f86854,
  0x07f8a3ea, 0x07f8ddae, 0x07f915ae, 0x07f94bf8,
  0x07f98099, 0x07f9b39e, 0x07f9e514, 0x07fa1507,
  0x07fa4381, 0x07fa7090, 0x07fa9c3e, 0x07fac696,
  0x07faefa2, 0x07fb176c, 0x07fb3dfe, 0x07fb6362,
  0x07fb87a0, 0x07fbaac3, 0x07fbccd1, 0x07fbedd5,
  0x07fc0dd6, 0x07fc2cdb, 0x07fc4aed, 0x07fc6813,
  0x07fc8454, 0x07fc9fb8, 0x07fcba44, 0x07fcd400,
  0x07fcecf2, 0x07fd0520, 0x07fd1c90, 0x07fd3348,
  0x07fd494e, 0x07fd5ea7, 0x07fd7357, 0x07fd8766,
  0x07fd9ad6, 0x07fdadae, 0x07fdbff2, 0x07fdd1a7,
  0x07fde2d0, 0x07fdf372, 0x07fe0392, 0x07fe1332,
  0x07fe2258, 0x07fe3107, 0x07fe3f42, 0x07fe4d0e,
  0x07fe5a6d, 0x07fe6763, 0x07fe73f3, 0x07fe8020,
  0x07fe8bed, 0x07fe975e, 0x07fea274, 0x07fead34,
  0x07feb79e, 0x07fec1b7, 0x07fecb81, 0x07fed4fd,
  0x07fede2f, 0x07fee718, 0x07feefbc, 0x07fef81b,
  0x07ff0038, 0x07ff0816, 0x07ff0fb6, 0x07ff171a,
  0x07ff1e43, 0x07ff2534, 0x07ff2bef, 0x07ff3275,
  0x07ff38c7, 0x07ff3ee7, 0x07ff44d8, 0x07ff4a99
};
    
// look-up table INVERSE LOGISTIC function
const net_t inv_sigmoid_lut[SPINN_SIGMD_RES] = 
{
  0x00000000, 0x000fffc5, 0x001fffaa, 0x002fffcf,
  0x00400055, 0x0050015a, 0x00600300, 0x00700565,
  0x008008ab, 0x00900cf2, 0x00a01259, 0x00b01901,
  0x00c0210a, 0x00d02a95, 0x00e035c2, 0x00f042b1,
  0x01005184, 0x0110625b, 0x01207556, 0x01308a97,
  0x0140a23f, 0x0150bc6f, 0x0160d947, 0x0170f8ea,
  0x01811b79, 0x01914116, 0x01a169e1, 0x01b195fe,
  0x01c1c58f, 0x01d1f8b5, 0x01e22f93, 0x01f26a4c,
  0x0202a903, 0x0212ebda, 0x022332f5, 0x02337e77,
  0x0243ce84, 0x02542340, 0x02647cce, 0x0274db54,
  0x02853ef6, 0x0295a7d8, 0x02a61620, 0x02b689f3,
  0x02c70377, 0x02d782d2, 0x02e80829, 0x02f893a5,
  0x0309256a, 0x0319bda2, 0x032a5c73, 0x033b0204,
  0x034bae80, 0x035c620d, 0x036d1cd5, 0x037ddf01,
  0x038ea8bc, 0x039f7a2f, 0x03b05385, 0x03c134e8,
  0x03d21e85, 0x03e31088, 0x03f40b1c, 0x04050e6f,
  0x04161aad, 0x04273005, 0x04384ea6, 0x044976bd,
  0x045aa87b, 0x046be40f, 0x047d29ab, 0x048e797e,
  0x049fd3bb, 0x04b13895, 0x04c2a83e, 0x04d422ea,
  0x04e5a8ce, 0x04f73a1d, 0x0508d70f, 0x051a7fd9,
  0x052c34b2, 0x053df5d3, 0x054fc375, 0x05619dd0,
  0x0573851f, 0x0585799d, 0x05977b86, 0x05a98b17,
  0x05bba88f, 0x05cdd42a, 0x05e00e2a, 0x05f256cf,
  0x0604ae5a, 0x0617150e, 0x06298b30, 0x063c1103,
  0x064ea6ce, 0x06614cd7, 0x06740368, 0x0686cac9,
  0x0699a346, 0x06ac8d2a, 0x06bf88c2, 0x06d2965e,
  0x06e5b64d, 0x06f8e8e0, 0x070c2e6b, 0x071f8742,
  0x0732f3bb, 0x0746742e, 0x075a08f3, 0x076db265,
  0x078170e2, 0x079544c7, 0x07a92e75, 0x07bd2e4d,
  0x07d144b4, 0x07e57210, 0x07f9b6c9, 0x080e1348,
  0x082287fb, 0x0837154f, 0x084bbbb6, 0x08607ba3,
  0x0875558c, 0x088a49e9, 0x089f5935, 0x08b483ed,
  0x08c9ca92, 0x08df2da8, 0x08f4adb3, 0x090a4b3e,
  0x092006d5, 0x0935e108, 0x094bda6a, 0x0961f390,
  0x09782d16, 0x098e8799, 0x09a503bc, 0x09bba222,
  0x09d26377, 0x09e94869, 0x0a0051a9, 0x0a177fef,
  0x0a2ed3f7, 0x0a464e81, 0x0a5df054, 0x0a75ba3a,
  0x0a8dad03, 0x0aa5c986, 0x0abe10a0, 0x0ad68331,
  0x0aef2223, 0x0b07ee65, 0x0b20e8ed, 0x0b3a12b8,
  0x0b536cc9, 0x0b6cf82f, 0x0b86b5fc, 0x0ba0a74e,
  0x0bbacd4a, 0x0bd5291e, 0x0befbc01, 0x0c0a8736,
  0x0c258c09, 0x0c40cbd0, 0x0c5c47ed, 0x0c7801ce,
  0x0c93faec, 0x0cb034ce, 0x0cccb109, 0x0ce9713f,
  0x0d06771f, 0x0d23c46b, 0x0d415af3, 0x0d5f3c98,
  0x0d7d6b4d, 0x0d9be918, 0x0dbab811, 0x0dd9da69,
  0x0df95261, 0x0e192254, 0x0e394cb6, 0x0e59d413,
  0x0e7abb11, 0x0e9c0474, 0x0ebdb31d, 0x0edfca0d,
  0x0f024c66, 0x0f253d6e, 0x0f48a092, 0x0f6c7964,
  0x0f90cba2, 0x0fb59b37, 0x0fdaec3f, 0x1000c307,
  0x10272414, 0x104e1425, 0x10759836, 0x109db586,
  0x10c6719d, 0x10efd24c, 0x1119ddb8, 0x11449a5d,
  0x11700f17, 0x119c4327, 0x11c93e3c, 0x11f7087c,
  0x1225aa90, 0x12552da8, 0x12859b91, 0x12b6feba,
  0x12e96246, 0x131cd21d, 0x13515aff, 0x13870a95,
  0x13bdef8d, 0x13f619b1, 0x142f9a07, 0x146a82ef,
  0x14a6e84f, 0x14e4dfb7, 0x1524809a, 0x1565e482,
  0x15a92756, 0x15ee67a3, 0x1635c6f9, 0x167f6a50,
  0x16cb7a83, 0x171a24e3, 0x176b9be2, 0x17c017e0,
  0x1817d824, 0x18732408, 0x18d24c69, 0x1935ad6e,
  0x199db0b6, 0x1a0ad024, 0x1a7d995a, 0x1af6b248,
  0x1b76df04, 0x1bff09a0, 0x1c904c9a, 0x1d2c0142,
  0x1dd3d3d5, 0x1e89e05e, 0x1f50dd65, 0x202c5d19,
  0x21213502, 0x22362b1e, 0x23752904, 0x24ed88ce,
  0x26b8f926, 0x290784d0, 0x2c44a1b6, 0x31c4649e
};

#endif
