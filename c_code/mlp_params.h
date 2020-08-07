#ifndef __MLP_PARAMS_H__
#define __MLP_PARAMS_H__

#include "limits.h"

// ------------------------------------------------------------------------
// MLP parameters
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// setup constants
// ------------------------------------------------------------------------
#define SPINN_TIMER_TICK_PERIOD  1000000
#define SPINN_PRINT_SHIFT        16

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
#define SPINN_DATA_KEY       0x00000000
#define SPINN_SYNC_KEY       0x00001000
#define SPINN_LDST_KEY       0x00002000
#define SPINN_LDSA_KEY       0x00003000
#define SPINN_LDSR_KEY       0x00004000
#define SPINN_CRIT_KEY       0x00005000
#define SPINN_STPN_KEY       0x00006000
#define SPINN_STOP_KEY       0x00007000

// packet type mask
#define SPINN_TYPE_MASK      0x0000f000

// packet condition keys
#define SPINN_PHASE_KEY(p)   (p << SPINN_PHASE_SHIFT)
#define SPINN_COLOUR_KEY     SPINN_COLOUR_MASK

// packet condition masks
#define SPINN_PHASE_SHIFT    11
#define SPINN_PHASE_MASK     (1 << SPINN_PHASE_SHIFT)
#define SPINN_COLOUR_SHIFT   10
#define SPINN_COLOUR_MASK    (1 << SPINN_COLOUR_SHIFT)

// block management
#define SPINN_BLOCK_SHIFT    5
#define SPINN_BLOCK_MASK     ((0xff << SPINN_BLOCK_SHIFT) & 0xff)
#define SPINN_BLOCK_KEY(p)   (p << SPINN_BLOCK_SHIFT)
#define SPINN_BLKOUT_MASK    ((1 << SPINN_BLOCK_SHIFT) - 1)
#define SPINN_BLKDLT_MASK    ((1 << SPINN_BLOCK_SHIFT) - 1)

// packet data masks
#define SPINN_OUTPUT_MASK    0x000000ff
#define SPINN_NET_MASK       0x000000ff
#define SPINN_DELTA_MASK     0x000000ff
#define SPINN_ERROR_MASK     0x000000ff
#define SPINN_STPD_MASK      0x000000ff
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// core function types
// ------------------------------------------------------------------------
#define SPINN_WEIGHT_PROC    0x0
#define SPINN_SUM_PROC       0x1
#define SPINN_THRESHOLD_PROC 0x2
#define SPINN_INPUT_PROC     0x3
#define SPINN_UNUSED_PROC    0x4
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
#define SPINN_THRD_PROC      1
#define SPINN_THRD_COMS      ((SPINN_THRD_PROC) << 1)
#define SPINN_THRD_STOP      ((SPINN_THRD_COMS) << 1)
#define SPINN_THRD_LDSA      ((SPINN_THRD_STOP) << 1)
#define SPINN_THRD_LDST      ((SPINN_THRD_LDSA) << 1)
#define SPINN_THRD_LDSR      (SPINN_THRD_LDSA)

#define SPINN_WF_THRDS       (SPINN_THRD_PROC | SPINN_THRD_COMS | SPINN_THRD_STOP)
#define SPINN_WB_THRDS       (SPINN_THRD_PROC)
#define SPINN_SF_THRDS       (SPINN_THRD_PROC | SPINN_THRD_STOP)
#define SPINN_SB_THRDS       (SPINN_THRD_PROC)
#define SPINN_IF_THRDS       (SPINN_THRD_PROC | SPINN_THRD_STOP)
#define SPINN_TF_THRDS       (SPINN_THRD_PROC | SPINN_THRD_STOP)
#define SPINN_TB_THRDS       (SPINN_THRD_PROC | SPINN_THRD_COMS)

// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// callback priorities
// ------------------------------------------------------------------------
// common non-queueable callbacks
#define SPINN_PACKET_P      -1
#define SPINN_TIMER_P        0

// weight core priorities
#define SPINN_WF_TICK_P      1
#define SPINN_WF_PROCESS_P   2
#define SPINN_WB_PROCESS_P   3

// sum core priorities
#define SPINN_S_PROCESS_P    1

// input core priorities
#define SPINN_I_PROCESS_P    1

// threshold core priorities
#define SPINN_TB_TICK_P      1
#define SPINN_TB_PROCESS_P   2
#define SPINN_TF_PROCESS_P   3

// stage exit function
#define SPINN_DONE_P         4
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// HOST communication commands
// ------------------------------------------------------------------------
// commands
// ------------------------------------------------------------------------
#define SPINN_HOST_FINAL     0
#define SPINN_HOST_NORMAL    1
#define SPINN_HOST_INFO      2


// ------------------------------------------------------------------------
// SDP parameters
// ------------------------------------------------------------------------
#define SPINN_SDP_IPTAG       2
#define SPINN_SDP_FLAGS       0x07
#define SPINN_SDP_TMOUT       100
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// EXIT codes -- error
// ------------------------------------------------------------------------
#define SPINN_NO_ERROR         0
#define SPINN_MEM_UNAVAIL      1
#define SPINN_QUEUE_FULL       2
#define SPINN_TIMEOUT_EXIT     3
#define SPINN_UNXPD_PKT        4
#define SPINN_CFG_UNAVAIL      5
// ------------------------------------------------------------------------

#endif
