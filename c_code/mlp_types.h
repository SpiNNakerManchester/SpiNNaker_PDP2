#ifndef __MLP_TYPES_H__
#define __MLP_TYPES_H__

#include "mlp_params.h"

enum MLPRegions {
  SYSTEM        =  0,
  NETWORK       =  1,
  CORE          =  2,
  INPUTS        =  3,
  TARGETS       =  4,
  EXAMPLE_SET   =  5,
  EXAMPLES      =  6,
  EVENTS        =  7,
  WEIGHTS       =  8,
  ROUTING       =  9,
  STAGE         = 10,
  REC_INFO      = 11
};

enum MLPRecordings {
  OUTPUTS      = 0,
  TEST_RESULTS = 1,
  TICK_DATA    = 2
};

// t cores can have more than one FWD key (due to partitions)
// i cores can have more than one BKP key (due to partitions)
enum MLPKeys {
  FWD  = 0,
  BKP  = 1,
  FDS  = 2,
  STP  = 3,
  LDS  = 4,
  FWDT = 5,
  BKPI = 5
};


// ------------------------------------------------------------------------
// weights and weight changes
// --------------------------
// weights are s16.15
// long weights are s48.15
//
// weight changes are s16.15
// long weight changes are s48.15
// ------------------------------------------------------------------------
typedef int       weight_t;         // connection weight
typedef long long long_weight_t;    // intermediate connection weight

typedef int       wchange_t;        // connection weight change
typedef long long long_wchange_t;   // intermediate connection weight change

#define SPINN_WEIGHT_SHIFT       15
#define SPINN_WEIGHT_MAX         ((weight_t)  (0xffff << SPINN_WEIGHT_SHIFT))
#define SPINN_WEIGHT_MIN         ((weight_t) -(0xffff << SPINN_WEIGHT_SHIFT))
#define SPINN_WEIGHT_POS_EPSILON ((weight_t)  1)
#define SPINN_WEIGHT_NEG_EPSILON ((weight_t) -1)
#define SPINN_WEIGHT_ONE         (1 << SPINN_WEIGHT_SHIFT)
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// activations
// --------------------------
// short activations are s0.15
// activations are s4.27
// long activations are s36.27
//NOTE: activation functions assume SPINN_ACTIV_SHIFT == SPINN_LONG_ACTIV_SHIFT
// ------------------------------------------------------------------------
typedef short     short_activ_t;
typedef int       activation_t;     // unit output or activation
typedef long long long_activ_t;     // intermediate unit output or activation

#define SPINN_SHORT_ACTIV_SIZE      16
#define SPINN_SHORT_ACTIV_SHIFT     15
#define SPINN_SHORT_ACTIV_MAX       SHRT_MAX
#define SPINN_SHORT_ACTIV_MIN       SHRT_MIN
// minimum positive value
#define SPINN_SHORT_ACTIV_MIN_POS   0

#define SPINN_ACTIV_SIZE            32
#define SPINN_ACTIV_SHIFT           27
// Not-a-Number used to represent undefined
#define SPINN_ACTIV_NaN             (1 << (SPINN_ACTIV_SIZE - 1))
// these values are set to compute the cross entropy error function
#define SPINN_ACTIV_ONE             (1 << SPINN_ACTIV_SHIFT)

#define SPINN_LONG_ACTIV_SHIFT      27
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// input/output derivatives
// --------------------------
// short derivatives are s0.15
// derivatives are s16.15
// long derivatives are s36.27
// ------------------------------------------------------------------------
typedef short     short_deriv_t;    // input or output derivative
typedef int       derivative_t;     // intermediate input/output derivative
typedef long long long_deriv_t;     // intermediate input/output derivative

#define SPINN_SHORT_DERIV_SHIFT     15
#define SPINN_SHORT_DERIV_MAX       SHRT_MAX
#define SPINN_SHORT_DERIV_MIN       SHRT_MIN

#define SPINN_DERIV_SHIFT           15
#define SPINN_DERIV_MAX             INT_MAX
#define SPINN_DERIV_MIN             INT_MIN
// these values are used to compute the cross entropy error
#define SPINN_DERIV_ONE             (1 << SPINN_DERIV_SHIFT)
#define SPINN_DERIV_NEG_ONE         (-1 << SPINN_DERIV_SHIFT)

#define SPINN_LONG_DERIV_SHIFT      27
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// nets
// --------------------------
// nets are s8.23
// long nets are s40.23
//NOTE: activation functions assume SPINN_NET_SHIFT == SPINN_LONG_NET_SHIFT
// ------------------------------------------------------------------------
typedef int       net_t;            // internal net (inputs dot-product)
typedef long long long_net_t;       // intermediate internal net

#define SPINN_NET_SHIFT             23
#define SPINN_NET_MAX               INT_MAX
#define SPINN_NET_MIN               INT_MIN

#define SPINN_LONG_NET_SHIFT        SPINN_NET_SHIFT
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// errors
// --------------------------
// errors are s16.15
// long errors are s36.27
// ------------------------------------------------------------------------
typedef int       error_t;          // output error
typedef long long long_error_t;     // intermediate output error

#define SPINN_ERROR_SHIFT           15
#define SPINN_ERROR_MAX             INT_MAX
#define SPINN_ERROR_MIN             INT_MIN

#define SPINN_LONG_ERR_SHIFT        27
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// deltas
// --------------------------
// deltas are s8.23
// long_deltas are s36.27
// ------------------------------------------------------------------------
typedef int       delta_t;          // error delta
typedef long long long_delta_t;     // intermediate error delta

#define SPINN_DELTA_SHIFT           23
#define SPINN_DELTA_MAX             INT_MAX
#define SPINN_DELTA_MIN             INT_MIN

#define SPINN_LONG_DELTA_SHIFT      27
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// link_delta sums
// --------------------------
// link delta sums are u28.4
// long link delta sums are u60.4
// ------------------------------------------------------------------------
typedef uint               lds_t;       // link delta sum
typedef unsigned long long long_lds_t;  // intermediate link delta sum

#define SPINN_LDS_SHIFT             4
#define SPINN_LDS_MAX               UINT_MAX
#define SPINN_LDS_MIN               0
#define SPINN_LDS_ONE               (1 << SPINN_LDS_SHIFT)

#define SPINN_LONG_LDS_SHIFT        4
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// fixed-point reals
// --------------------------
// short fixed-point reals are s0.15
// fixed-point reals are s15.16
// long fixed-point reals are s47.16
//NOTE: may be a good idea to change fpreal to s16.15 for compatibility!
// ------------------------------------------------------------------------
typedef short     short_fpreal;
typedef int       fpreal;
typedef long long long_fpreal;

#define SPINN_SHORT_FPREAL_SHIFT    15

#define SPINN_FPREAL_SIZE           32
#define SPINN_FPREAL_SHIFT          16
#define SPINN_FP_NaN                (1 << (SPINN_FPREAL_SIZE - 1))
#define SPINN_SMALL_VAL             1

#define SPINN_LONG_FPREAL_SHIFT     16
// ------------------------------------------------------------------------


typedef uint scoreboard_t;      // keep track of received items

typedef uchar proc_phase_t;     // phase (FORWARD or BACKPROP)


// ------------------------------------------------------------------------
// network configuration
// ------------------------------------------------------------------------
typedef struct network_conf     // MLP network configuration
{
  uchar net_type;               // type of neural net
  uint  ticks_per_int;          // number of ticks per interval
  uint  global_max_ticks;       // max number of ticks across all the examples
  uint  num_write_blks;         // number of groups that write outputs
} network_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// weight core configuration
// ------------------------------------------------------------------------
// The neural net is represented by a weight matrix.
// The matrix is divided into num_rblks x num_cblk weight blocks
// and every weight core computes for one of these blocks.
// Each block is associated with a single projection, i.e., it contains
// connection weights associated with a single origin group and a single
// destination group (which can be the same in recurrent networks).
// Weights are usually associated with the destination group.
// weight cores compute unit net (FORWARD phase) and error (BACKPROP phase)
// block dot-products (b-d-p) and weight updates.
// ------------------------------------------------------------------------
typedef struct w_conf             // weight core configuration
{
  uint         num_rows;          // rows in this core's block
  uint         num_cols;          // columns in this core's block
  uint         row_blk;           // this core's row block number
  uint         col_blk;           // this core's column block number
  scoreboard_t sync_expected;     // num of expected sync packets
  activation_t initOutput;        // initial value for unit outputs
  short_fpreal learningRate;      // network learning rate
  short_fpreal weightDecay;       // network weight decay
  short_fpreal momentum;          // network momentum
} w_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// sum core configuration
// ------------------------------------------------------------------------
// sum cores accumulate accumulate b-d-ps sent by weight cores and
// compute unit nets (FORWARD phase) and errors (BACKPROP phase)
// ------------------------------------------------------------------------
typedef struct s_conf               // sum core configuration
{
  uint         num_units;           // this core's number of units
  scoreboard_t fwd_expected;        // num of expected partial nets
  scoreboard_t bkp_expected;        // num of expected partial errors
  scoreboard_t ldsa_expected;       // num of expected partial link delta sums
  scoreboard_t ldst_expected;       // num of expected link delta sum totals
  uchar        is_first_group;      // is this the first group in the network?
} s_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// input core configuration
// ------------------------------------------------------------------------
// input cores process the values received from the sum cores through the
// elements of the input pipeline as required by LENS.
// ------------------------------------------------------------------------
typedef struct i_conf                // input core configuration
{
  uchar         output_grp;          // is this an OUTPUT group?
  uchar         input_grp;           // is this an INPUT group?
  uint          num_units;           // this core's number of units
  uint          partitions;          // this groups's number of partitions
  uint          num_in_procs;        // number of input (net) comp procedures
  uint          procs_list[SPINN_NUM_IN_PROCS];
  uchar         in_integr_en;        // input INTEGRATOR in use
  fpreal        in_integr_dt;        // integration time const for input integr
  fpreal        soft_clamp_strength; // Strength coeff for soft clamp
  net_t         initNets;            // initial value for unit nets
  activation_t  initOutput;          // initial value for unit outputs
} i_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// threshold core configuration
// ------------------------------------------------------------------------
// In the FORWARD phase, threshold cores compute the activation or unit
// output applying an activation function to the nets sent by the input cores.
// External inputs are also processed by threshold cores in this phase.
// In the BACKPROP phase, these cores compute the error deltas from the
// errors computed locally (output groups only) or sent by the sum cores.
// ------------------------------------------------------------------------
typedef struct t_conf                  // threshold core configuration
{
  uchar         output_grp;            // is this an OUTPUT group?
  uchar         input_grp;             // is this an INPUT group?
  uint          num_units;             // this core's number of units
  uint          partitions;            // this group's number of partitions
  uchar         write_results;         // record test results?
  uchar         write_out;             // record outputs?
  uchar         last_tick_only;        // record only last tick of examples?
  uint          write_blk;             // this core's write block
  uchar         hard_clamp_en;         // HARD CLAMP in use
  uchar         out_integr_en;         // output INTEGRATOR in use
  fpreal        out_integr_dt;         // integration time const for input integr
  uint          num_out_procs;         // number of output comp procedures
  uint          procs_list[SPINN_NUM_OUT_PROCS];
  fpreal        weak_clamp_strength;   // Strength coeff for weak clamp
  activation_t  initOutput;            // initial value for unit outputs
  error_t       tst_group_criterion;   // test-mode convergence criterion value
  error_t       trn_group_criterion;   // train-mode convergence criterion value
  uchar         criterion_function;    // function to eval convergence criterion
  uchar         is_first_output_group; // is this the first of the output groups
  uchar         is_last_output_group;  // is this the last of the output groups
  uchar         error_function;        // error function used for BACKPROP
} t_conf_t;
// ------------------------------------------------------------------------


typedef struct
{
  uint key;               // packet key (for routing)
  uint payload;           // packet payload (optional)
} packet_t;


typedef struct
{
  // enqueue to tail, dequeue from head
  volatile uint head;     // pointer to queue start
  volatile uint tail;     // pointer to queue end
  packet_t *    queue;    // pointer to actual queue
} pkt_queue_t;

#endif


// ------------------------------------------------------------------------
// example set, example and event configurations
// ------------------------------------------------------------------------
// examples are organised as LENS-style examples
// ------------------------------------------------------------------------
typedef struct mlp_set
{
  uint    num_examples;
  fpreal  max_time;
  fpreal  min_time;
  fpreal  grace_time;
} mlp_set_t;


typedef struct mlp_example
{
  uint   num;
  uint   num_events;
  uint   ev_idx;
  fpreal freq;
} mlp_example_t;


typedef struct mlp_event
{
  fpreal  max_time;
  fpreal  min_time;
  fpreal  grace_time;
  uint    it_idx;
} mlp_event_t;


// ------------------------------------------------------------------------
// stage configuration
// ------------------------------------------------------------------------
typedef struct stage_conf       // execution stage configuration
{
  uchar stage_id;               // stage number
  uchar training;               // stage mode: train (1) or test (0)
  uchar update_function;        // weight update function in this stage
  uchar reset;                  // reset example index at stage start?
  uint  num_examples;           // number of examples to run in this stage
  uint  num_epochs;             // number of training epochs in this stage
} stage_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// tick record
// ------------------------------------------------------------------------
typedef struct tick_record {
  uint epoch;    // current epoch
  uint example;  // current example
  uint event;    // current event
  uint tick;     // current tick
} tick_record_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// test results record
// ------------------------------------------------------------------------
typedef struct test_results {
  uint epochs_trained;    // epochs of training that have occurred
  uint examples_tested;   // total number of examples tested
  uint ticks_tested;      // total number of ticks in those examples
  uint examples_correct;  // number of examples that met the testing criterion
} test_results_t;
// ------------------------------------------------------------------------


typedef void (*out_proc_t) (uint);   // output comp procedures


typedef void (*out_proc_back_t) (uint);   // BACKPROP output comp procedures


typedef uint (*out_proc_init_t) (void);    // input initialisation procedures


typedef void (*in_proc_t) (uint);    // input (net) comp procedures


typedef void (*in_proc_back_t) (uint);    // BACKPROP input (net) comp procedures


typedef uint (*in_proc_init_t) (void);    // input initialisation procedures


typedef void (*stop_crit_t) (uint);  // stopping criterion comp procedures


typedef void (*out_error_t) (uint);   // error comp procedures


typedef void (*weight_update_t) (void);   // weight update procedures
