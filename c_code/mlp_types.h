#ifndef __MLP_TYPES_H__
#define __MLP_TYPES_H__

#include "mlp_params.h"

enum MLPRegions {
	NETWORK,
	CORE,
	INPUTS,
	TARGETS,
	EXAMPLE_SET,
	EXAMPLES,
	EVENTS,
	WEIGHTS,
	ROUTING
};

enum MLPKeys {
	FWD,
	BKP,
	FDS,
	STP
};

typedef short     short_activ_t;    // unit output or activation
typedef int       activation_t;     // intermediate unit output or activation
typedef long long long_activ_t;     // intermediate unit output or activation

// short activations are s0.15
#define SPINN_SHORT_ACTIV_SHIFT  15
#define SPINN_SHORT_ACTIV_MAX    ((1 << SPINN_SHORT_ACTIV_SHIFT) - 1)
#define SPINN_SHORT_ACTIV_MIN    0
// minimum negative value for an activation variable
#define SPINN_SHORT_ACTIV_MIN_NEG      (-1 * SPINN_SHORT_ACTIV_MAX)
#define SPINN_SHORT_ACTIV_NaN          (-1 << SPINN_SHORT_ACTIV_SHIFT)

// activations are s4.27
#define SPINN_ACTIV_SHIFT        27
#define SPINN_ACTIV_MAX          INT_MAX
#define SPINN_ACTIV_MIN          0
#define SPINN_ACTIV_NaN          (-1 << SPINN_ACTIV_SHIFT)
// minimum negative value for a long activation variable
//~#define SPINN_ACTIV_MIN_NEG   INT_MIN
// these values are set to compute the cross entropy error function
#define SPINN_ACTIV_ONE          (1 << SPINN_ACTIV_SHIFT)
//~#define SPINN_ACTIV_NEG_ONE   (-1 << SPINN_ACTIV_SHIFT)

typedef short     short_deriv_t;  // input or output derivative
typedef int       derivative_t;   // intermediate unit input or output derivative
typedef long long long_deriv_t;   // intermediate unit input or output derivative

// short derivatives are s0.15
#define SPINN_DERIV_SHIFT        15
#define SPINN_SHORT_DERIV_MAX         ((1 << SPINN_DERIV_SHIFT) - 1)
//~#define SPINN_SHORT_DERIV_MIN          0
// minimum negative value for an derivative variable
#define SPINN_SHORT_DERIV_MIN_NEG     (-1 * SPINN_SHORT_DERIV_MAX)
#define SPINN_SHORT_DERIV_NaN          (-1 << SPINN_DERIV_SHIFT)

// derivatives are s16.15
#define SPINN_DERIV_MAX          INT_MAX
#define SPINN_DERIV_MIN          0
// minimum negative value for a long derivative variable
#define SPINN_DERIV_MIN_NEG      INT_MIN
// these values are set to compute the cross entropy error function
#define SPINN_DERIV_ONE          (1 << SPINN_DERIV_SHIFT)
#define SPINN_DERIV_NEG_ONE      (-1 << SPINN_DERIV_SHIFT)

// long derivatives are s36.27
#define SPINN_LONG_DERIV_SHIFT  27
//~#define SPINN_LONG_DERIV_MAX     SPINN_LONG_ACTIV_MAX
//~#define SPINN_LONG_DERIV_MIN     SPINN_LONG_ACTIV_MIN
// minimum negative value for a long long derivative variable
//~#define SPINN_LONG_DERIV_MIN_NEG   LONG_MIN

typedef int       net_t;            // unit internal net (inputs dot-product)
typedef long long long_net_t;       // used for net intermediate calc

//TODO: set these values correctly!
// nets are s8.23
#define SPINN_NET_SHIFT          23
#define SPINN_NET_MAX            ( 255.0 * (1 << SPINN_NET_SHIFT))
#define SPINN_NET_MIN            (-255.0 * (1 << SPINN_NET_SHIFT))

// long nets are s40.23

typedef int       error_t;          // unit output error
typedef long long long_error_t;     // used for error intermediate calc

//TODO: set these values correctly!
// errors are s16.15
#define SPINN_ERROR_SHIFT        15
#define SPINN_ERROR_MAX          (  0xffff * (1 << SPINN_ERROR_SHIFT))
#define SPINN_ERROR_MIN          (-(0xffff * (1 << SPINN_ERROR_SHIFT)))

// long errors are s36.27
#define SPINN_LONG_ERR_SHIFT     27
//~#define SPINN_LONG_ERR_MAX       (  0xffff * (1 << SPINN_LONG_ERR_SHIFT))
//~#define SPINN_LONG_ERR_MIN       (-(0xffff * (1 << SPINN_LONG_ERR_SHIFT)))

typedef int       delta_t;          // input derivative
typedef long long long_delta_t;     // used for delta intermediate calc

// deltas are s16.15
//!#define SPINN_DELTA_SHIFT        15
//!#define SPINN_DELTA_MAX          (  0xffff * (1 << SPINN_DELTA_SHIFT))
//!#define SPINN_DELTA_MIN          (-(0xffff * (1 << SPINN_DELTA_SHIFT)))
// deltas are s8.23
#define SPINN_DELTA_SHIFT        23
#define SPINN_DELTA_MAX          (  0xff * (1 << SPINN_DELTA_SHIFT))
#define SPINN_DELTA_MIN          (-(0xff * (1 << SPINN_DELTA_SHIFT)))

// long_deltas are s36.27
#define SPINN_LONG_DELTA_SHIFT   27
//~#define SPINN_LONG_DELTA_MAX     SPINN_LONG_ERR_MAX
//~#define SPINN_LONG_DELTA_MIN     SPINN_LONG_ERR_MIN

// weights are s16.15
// long weights are s48.15
// weight changes are s16.15
// long weight changes are s48.15
typedef int       weight_t;         // connection weight
typedef long long long_weight_t;    // intermediate conntection weight
typedef int       wchange_t;        // connection weight change
typedef long long long_wchange_t;   // intermediate connection weight change

#define SPINN_WEIGHT_SHIFT     15
#define SPINN_WEIGHT_MAX       ((weight_t)  (0xffff << SPINN_WEIGHT_SHIFT))
#define SPINN_WEIGHT_MIN       ((weight_t) -(0xffff << SPINN_WEIGHT_SHIFT))
#define SPINN_WEIGHT_POS_DELTA ((weight_t)  1)
#define SPINN_WEIGHT_NEG_DELTA ((weight_t) -1)

typedef int       fpreal;           // 32-bit fixed-point number
typedef long long lfpreal;          // 64-bit fixed-point number

//NOTE: may be a good idea to change to s16.15 for compatibility!
// fixed-point reals are s15.16
// long fixed-point reals are s47.16
#define SPINN_FPREAL_SHIFT       16
#define SPINN_FP_NaN             (-1 << SPINN_FPREAL_SHIFT)
#define SPINN_SMALL_VAL          1

typedef uint      scoreboard_t;     // keep track of received items

typedef uchar     proc_phase_t;     // phase (FORWARD or BACKPROP)


// ------------------------------------------------------------------------
// network configuration
// ------------------------------------------------------------------------
typedef struct network_conf     // MLP network configuration
{
  uchar net_type;               // type of neural net
  uchar training;               // training or testing mode?
  uint  num_epochs;             // number of epochs to run
  uint  num_examples;           // number of examples per epoch
  uint  ticks_per_int;          // number of ticks per interval
  uint  global_max_ticks;       // max number of ticks across all the examples
  uint  num_write_blks;         // number of groups that write outputs
  uint  timeout;                // in case something goes wrong
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
typedef struct w_conf               // weight core configuration
{
  uchar        to_out_grp;          // projection weights into an OUTPUT group?
  uchar        from_out_grp;        // projection weights from an OUTPUT group?
  uchar        to_input_grp;        // projection weights into an INPUT group?
  uchar        from_input_grp;      // projection weights from an INPUT group?
  uint         num_rblks;           // blocks in a row of the global matrix
  uint         num_cblks;           // blocks in a column of the global matrix
  uint         blk_row;             // this core's block row coordinate
  uint         blk_col;             // this core's block column coordinate
  uint         num_rows;            // rows in this core's block
  uint         num_cols;            // columns in this core's block
  scoreboard_t f_all_arrived;       // all expected unit outputs
  scoreboard_t b_all_arrived;       // all expected error deltas
  short_activ_t learningRate;       // network learning rate
  uint         weights_struct_addr; // address in SDRAM for weight file
} w_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// sum core configuration
// ------------------------------------------------------------------------
// sum cores accumulate acummulate b-d-ps sent by weight cores and
// compute unit nets (FORWARD phase) and errors (BACKPROP phase)
// ------------------------------------------------------------------------
typedef struct s_conf               // sum core configuration
{
  uint         num_nets;            // this core's number of unit nets
  scoreboard_t all_arrived;         // all expected partial nets/errors
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
  uint          num_nets;            // this core's number of unit nets
  uint          num_in_procs;        // number of input (net) comp procedures
  uint          procs_list[SPINN_NUM_IN_PROCS];
  uchar         in_integr_en;        // input integrator in use
  fpreal        in_integr_dt;        // integration time const for input integr
  fpreal        soft_clamp_strength; // Strength coeff for soft clamp fix 16.16
  net_t         initNets;            // initial value for unit nets
  short_activ_t initOutput;          // initial value for unit outputs
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
typedef struct t_conf    // threshold core configuration
{
  uchar        output_grp;            // is this an OUTPUT group?
  uchar        input_grp;             // is this an INPUT group?
  uint         num_outputs;           // this core's number of unit outputs
  uint         num_ext_inputs;        // this core's number of external inputs
  uint         output_offset;         // offset wrt to global outputs
  uint         output_blk;            // this core's unit output block
  uint         delta_blk;             // this core's delta block
  scoreboard_t f_all_arrived;         // all expected nets
  scoreboard_t b_all_arrived;         // all expected errors
  scoreboard_t f_s_all_arr;           // all expected FORWARD sync packets
  scoreboard_t b_s_all_arr;           // all expected BACKPROP sync packets
  uchar        write_out;             // write outputs (send to host)?
  uint         write_blk;             // this core's write block
//   uint         example_set_addr;      // addr in SDRAM for example set file
//   uint         examples_addr;         // addr in SDRAM for examples file
//   uint         events_addr;           // addr in SDRAM for events file
  uint         inputs_addr;           // address in SDRAM for inputs file
  uint         targets_addr;          // address in SDRAM for targets file
  uint         num_in_procs;          // number of input comp procs - used in
                                      // output initialization
  uchar        out_integr_en;         // input integrator in use
  fpreal       out_integr_dt;         // integration time const for input integr
  uint         num_out_procs;         // number of output comp procedures
  uint         procs_list[SPINN_NUM_OUT_PROCS];
  fpreal       weak_clamp_strength;   // Strength coeff for weak clamp fix 16.16
  short_activ_t initOutput;           // initial value for unit outputs
  error_t      group_criterion;       // convergence criterion value
  uchar        criterion_function;    // function to eval convergence criterion
  uchar        is_first_output_group; // is this the firso of the output groups
  uchar        is_last_output_group;  // is this the last of the output groups
  uchar        error_function;        // is the error function id to be used for
                                      // BACKPROP
  uint         group_id;              // ID of the group
  uint         subgroup_id;           // ID of the subgroup
  uint         total_subgroups;       // total number of subgroups
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
// example configuration in SDRAM
// ------------------------------------------------------------------------
// Examples are organized as LENS examples (see LENS documentation).
// ------------------------------------------------------------------------
typedef struct mlp_set {
  uint   num_examples;
  fpreal  max_time;
  fpreal  min_time;
  fpreal  grace_time;
} mlp_set_t;


typedef struct mlp_example {
  uint   num;
  uint   num_events;
  uint   ev_idx;
  fpreal freq;
} mlp_example_t;


typedef struct mlp_event {
  fpreal  max_time;
  fpreal  min_time;
  fpreal  grace_time;
  uint    it_idx;
} mlp_event_t;


typedef void (*out_proc_t) (uint);   // output comp procedures


typedef void (*out_proc_back_t) (uint);   // BACKPROP output comp procedures


typedef int  (*out_proc_init_t) (void);    // input initialization procedures


typedef void (*in_proc_t) (uint);    // input (net) comp procedures


typedef void (*in_proc_back_t) (uint);    // BACKPROP input (net) comp procedures


typedef int  (*in_proc_init_t) (void);    // input initialization procedures


typedef void (*stop_crit_t) (uint);  // stopping criterion comp procedures


typedef void (*out_error_t) (uint);   // error comp procedures
