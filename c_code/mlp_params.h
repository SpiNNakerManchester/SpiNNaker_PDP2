#ifndef __MLP_PARAMS_H__
#define __MLP_PARAMS_H__

#include "limits.h"

// ------------------------------------------------------------------------
// MLP parameters
// ------------------------------------------------------------------------
// software configuration
// ------------------------------------------------------------------------
#define SPINN_USE_COUNTER_SB     TRUE
#define SPINN_WEIGHT_HISTORY     FALSE
#define SPINN_OUTPUT_HISTORY     FALSE


// ------------------------------------------------------------------------
// simulation constants
// ------------------------------------------------------------------------
#define SPINN_TIMER_TICK_PERIOD  1000000
//#define SPINN_TIMER_TICK_PERIOD  100000
#define SPINN_PRINT_DLY          200
#define SPINN_PRINT_SHIFT        16
#define SPINN_SKEW_DELAY         (chipID << 18) | (coreID << 16)
#define SPINN_TIMER2_DIV         10
#define SPINN_TIMER2_CONF        0x83
#define SPINN_TIMER2_LOAD        0


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
#define SPINN_BACKPROP      (!SPINN_FORWARD)

#define SPINN_W_INIT_TICK   0
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
#define SPINN_BLK_R_MASK     0xff000000
#define SPINN_BLK_C_MASK     0x00ff0000
#define SPINN_SUM_B_MASK     0x00ff0000
#define SPINN_THD_B_MASK     0x00ff0000
#define SPINN_STOP_MASK      0x00003a00
#define SPINN_STPF_MASK      0x00000800
#define SPINN_SYNC_MASK      0x00001000
#define SPINN_PHASE_MASK     0x00000800
#define SPINN_COLOUR_MASK    0x00000400
#define SPINN_FUNCT_MASK     0x00000300
#define SPINN_OUTPUT_MASK    0x000000ff
#define SPINN_NET_MASK       0x000000ff
#define SPINN_DELTA_MASK     0x000000ff
#define SPINN_ERROR_MASK     0x000000ff
#define SPINN_STPD_MASK      0x000000ff

#define SPINN_BLK_R_SHIFT    24
#define SPINN_BLK_C_SHIFT    16
#define SPINN_SUM_B_SHIFT    16
#define SPINN_THD_B_SHIFT    16
#define SPINN_STOP_SHIFT     13
#define SPINN_SYNC_SHIFT     12
#define SPINN_PHASE_SHIFT    11
#define SPINN_COLOUR_SHIFT   10
#define SPINN_FUNCT_SHIFT    8
#define SPINN_STPD_SHIFT     0

#define SPINN_BR_KEY(br)     (br << SPINN_BLK_R_SHIFT)
#define SPINN_BC_KEY(bc)     (bc << SPINN_BLK_C_SHIFT)
#define SPINN_SB_KEY(bc)     (bc << SPINN_SUM_B_SHIFT)
#define SPINN_TB_KEY(bc)     (bc << SPINN_THD_B_SHIFT)
#define SPINN_PHASE_KEY(p)   (p << SPINN_PHASE_SHIFT)
#define SPINN_FUNCT_KEY(f)   (f << SPINN_FUNCT_SHIFT)
#define SPINN_COLOUR_KEY     SPINN_COLOUR_MASK
#define SPINN_CORETYPE_KEY   (coreType << SPINN_FUNCT_SHIFT)
#define SPINN_SYNC_KEY       SPINN_SYNC_MASK
#define SPINN_STPF_KEY       (SPINN_STOP_MASK & ~SPINN_STPF_MASK)
#define SPINN_STPR_KEY       SPINN_STOP_MASK
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
// implementation params
// ------------------------------------------------------------------------
//TODO: check if size is appropriate
#define SPINN_THLD_PQ_LEN    256
#define SPINN_WEIGHT_PQ_LEN  512
#define SPINN_SUM_PQ_LEN     512
#define SPINN_INPUT_PQ_LEN   512
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// callback priorities
// ------------------------------------------------------------------------
// non-queueable callbacks
#define SPINN_PACKET_P       -1
#define SPINN_TIMER_P        0

// queueable callbacks
#define SPINN_UPDT_WEIGHT_P  1
#define SPINN_WF_TICK_P      1
#define SPINN_WB_TICK_P      1
#define SPINN_WF_PROCESS_P   2
#define SPINN_WB_PROCESS_P   2

#define SPINN_S_TICK_P       1
#define SPINN_S_PROCESS_P    2

#define SPINN_I_TICK_P       1
#define SPINN_I_PROCESS_P    2

//TODO: review priorities
#define SPINN_T_INIT_OUT_P   1
#define SPINN_SEND_OUTS_P    1
#define SPINN_SEND_DELTAS_P  1
#define SPINN_SEND_STOP_P    1
#define SPINN_TF_TICK_P      1
#define SPINN_TB_TICK_P      1
#define SPINN_TF_PROCESS_P   2
#define SPINN_TB_PROCESS_P   2
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
#define SPINN_SDP_IPTAG       1
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
#define SPINN_CORE_TYPE_ERROR  5
// ------------------------------------------------------------------------

#endif
