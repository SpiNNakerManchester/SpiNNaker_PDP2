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

#ifndef __MLP_PARAMS_H__
#define __MLP_PARAMS_H__

#include "limits.h"

// ------------------------------------------------------------------------
// MLP parameters
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// setup constants
// ------------------------------------------------------------------------
#define SPINN_TIMER_TICK_PERIOD  100000
#define SPINN_PRINT_SHIFT        16
//NOTE: must be a power of 2!
#define SPINN_MAX_UNITS          256

// deadlock recovery constants
#define SPINN_DLRV_MAX_CNT       3

// boolean results can be transmitted inside SpiNNaker routing keys
#define SPINN_BOOL_ZERO      0
#define SPINN_BOOL_ONE       (1 << SPINN_BOOL_SHIFT)
#define SPINN_DLRV_ABRT      SPINN_BOOL_ONE
#define SPINN_RESTART        TRUE

// ------------------------------------------------------------------------
// profiler constants
// ------------------------------------------------------------------------
// configure timer2 for profiling: enabled, free running,
// interrupt disabled, no pre-scale and 32-bit one-shot mode
#define SPINN_PROFILER_CFG       0x83
#define SPINN_PROFILER_START     0xffffffff


// ------------------------------------------------------------------------
// neural net constants
// ------------------------------------------------------------------------
#define SPINN_NET_FEED_FWD       0
#define SPINN_NET_SIMPLE_REC     1
#define SPINN_NET_RBPTT          2
#define SPINN_NET_CONT           3


#define SPINN_NUM_IN_PROCS       2
//--------------------------
#define SPINN_IN_INTEGR          0
#define SPINN_IN_SOFT_CLAMP      1


#define SPINN_NUM_OUT_PROCS      5
//--------------------------
#define SPINN_OUT_LOGISTIC       0
#define SPINN_OUT_INTEGR         1
#define SPINN_OUT_HARD_CLAMP     2
#define SPINN_OUT_WEAK_CLAMP     3
#define SPINN_OUT_BIAS           4


#define SPINN_NUM_STOP_PROCS     3
//--------------------------
#define SPINN_NO_STOP            0
#define SPINN_STOP_STD           1
#define SPINN_STOP_MAX           2


#define SPINN_NUM_ERROR_PROCS    3
//--------------------------
#define SPINN_NO_ERR_FUNCTION    0
#define SPINN_ERR_CROSS_ENTROPY  1
#define SPINN_ERR_SQUARED        2


#define SPINN_NUM_UPDATE_PROCS   3
//--------------------------
#define SPINN_STEEPEST_UPDATE       0
#define SPINN_MOMENTUM_UPDATE       1
#define SPINN_DOUGSMOMENTUM_UPDATE  2


// ------------------------------------------------------------------------
// activation function options
// ------------------------------------------------------------------------
// input truncation is the default!
//#define SPINN_SIGMD_ROUNDI
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// phase or direction
// ------------------------------------------------------------------------
#define SPINN_FORWARD       0
#define SPINN_BACKPROP      1

#define SPINN_W_INIT_TICK   1
#define SPINN_S_INIT_TICK   1
#define SPINN_I_INIT_TICK   1
#define SPINN_T_INIT_TICK   1

#define SPINN_WB_END_TICK   1
#define SPINN_SB_END_TICK   1
#define SPINN_IB_END_TICK   1
#define SPINN_TB_END_TICK   1
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// multicast packet routing keys and masks
// ------------------------------------------------------------------------
// packet type keys
#define SPINN_KEY_SHIFT      12
#define SPINN_KEY_BITS       4
#define SPINN_TYPE_MASK      (((1 << SPINN_KEY_BITS) - 1) << SPINN_KEY_SHIFT)
#define SPINN_DATA_KEY       ( 0 << SPINN_KEY_SHIFT)
#define SPINN_SYNC_KEY       ( 1 << SPINN_KEY_SHIFT)
#define SPINN_FSGN_KEY       ( 2 << SPINN_KEY_SHIFT)
#define SPINN_BSGN_KEY       ( 3 << SPINN_KEY_SHIFT)
#define SPINN_LDSA_KEY       ( 4 << SPINN_KEY_SHIFT)
#define SPINN_CRIT_KEY       ( 5 << SPINN_KEY_SHIFT)
#define SPINN_STPN_KEY       ( 6 << SPINN_KEY_SHIFT)
#define SPINN_STOP_KEY       ( 7 << SPINN_KEY_SHIFT)
#define SPINN_DLRV_KEY       (15 << SPINN_KEY_SHIFT)

// packet condition keys
#define SPINN_PHASE_KEY(p)   (p << SPINN_PHASE_SHIFT)

// computation phase and colour
#define SPINN_PHASE_SHIFT    (SPINN_KEY_SHIFT - 1)
#define SPINN_PHASE_MASK     (1 << SPINN_PHASE_SHIFT)
#define SPINN_COLR_SHIFT     (SPINN_PHASE_SHIFT - 1)
#define SPINN_COLR_MASK      (1 << SPINN_COLR_SHIFT)

// boolean result (criterion, tick stop, abort and such)
#define SPINN_BOOL_SHIFT     (SPINN_COLR_SHIFT - 1)
#define SPINN_BOOL_MASK      (1 << SPINN_BOOL_SHIFT)
#define SPINN_CRIT_MASK      SPINN_BOOL_MASK
#define SPINN_STOP_MASK      SPINN_BOOL_MASK
#define SPINN_ABRT_MASK      SPINN_BOOL_MASK

// packet data masks
#define SPINN_TICK_MASK      (SPINN_BOOL_MASK - 1)
#define SPINN_OUTPUT_MASK    (SPINN_MAX_UNITS - 1)
#define SPINN_NET_MASK       (SPINN_MAX_UNITS - 1)
#define SPINN_DELTA_MASK     (SPINN_MAX_UNITS - 1)
#define SPINN_ERROR_MASK     (SPINN_MAX_UNITS - 1)
#define SPINN_STPD_MASK      (SPINN_MAX_UNITS - 1)
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// implementation parameters
// ------------------------------------------------------------------------
//TODO: check if sizes are appropriate
#define SPINN_THLD_PQ_LEN    256
#define SPINN_WEIGHT_PQ_LEN  512
#define SPINN_SUM_PQ_LEN     2048
#define SPINN_INPUT_PQ_LEN   512
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// thread parameters
// ------------------------------------------------------------------------
#define SPINN_NO_THRDS       0
#define SPINN_THRD_PROC      0x00000001
#define SPINN_THRD_COMS      0x00000002
#define SPINN_THRD_CRIT      0x00000004
#define SPINN_THRD_STOP      0x00000008
#define SPINN_THRD_LDSA      0x00000010
#define SPINN_THRD_FSGN      0x00000020
#define SPINN_THRD_BSGN      0x00000040

#define SPINN_WF_THRDS       (SPINN_NO_THRDS)
#define SPINN_WB_THRDS       (SPINN_NO_THRDS)
#define SPINN_SF_THRDS       (SPINN_NO_THRDS)
#define SPINN_SB_THRDS       (SPINN_THRD_PROC | SPINN_THRD_BSGN)
#define SPINN_IF_THRDS       (SPINN_NO_THRDS)
#define SPINN_IB_THRDS       (SPINN_NO_THRDS)
#define SPINN_TF_THRDS       (SPINN_THRD_PROC | SPINN_THRD_CRIT)
#define SPINN_TB_THRDS       (SPINN_THRD_COMS | SPINN_THRD_BSGN)

// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// callback priorities
// ------------------------------------------------------------------------
// common non-queueable callbacks
#define SPINN_PACKET_P      -1
#define SPINN_TIMER_P        0

// weight core priorities
#define SPINN_W_SEND_P       0
#define SPINN_W_TICK_P       1
#define SPINN_WF_PROCESS_P   2
#define SPINN_WB_PROCESS_P   3

// sum core priorities
#define SPINN_S_SEND_P       0
#define SPINN_S_TICK_P       1
#define SPINN_S_PROCESS_P    2

// input core priorities
#define SPINN_I_TICK_P       1
#define SPINN_I_PROCESS_P    2

// threshold core priorities
#define SPINN_T_SEND_P       0
#define SPINN_T_TICK_P       1
#define SPINN_TB_PROCESS_P   2
#define SPINN_TF_PROCESS_P   3

// stage exit function
#define SPINN_DONE_P         4
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// recording channels
// ------------------------------------------------------------------------
#define SPINN_REC_OUTPUTS    0
#define SPINN_REC_RESULTS    1
#define SPINN_REC_TICK_DATA  2


// ------------------------------------------------------------------------
// EXIT codes -- error
// ------------------------------------------------------------------------
#define SPINN_NO_ERROR       0
#define SPINN_MEM_UNAVAIL    1
#define SPINN_QUEUE_FULL     2
#define SPINN_TIMEOUT_EXIT   3
#define SPINN_UNXPD_PKT      4
#define SPINN_CFG_UNAVAIL    5
// ------------------------------------------------------------------------

#endif
