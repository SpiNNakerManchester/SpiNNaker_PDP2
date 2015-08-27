#ifndef __MLP_TYPES_H__
#define __MLP_TYPES_H__

#include "mlp_params.h"

typedef short     activation_t;     // unit output or activation (16 bit)
typedef int       long_activ_t;     // intermediate unit output or activation (32 bit)
typedef long long llong_activ_t;    // intermediate unit output or activation (64 bit)

// activations are 16-bit quantities with 15 decimal bits
#define SPINN_ACTIV_SHIFT        15
#define SPINN_ACTIV_MAX          ((1 << SPINN_ACTIV_SHIFT) - 1)
#define SPINN_ACTIV_MIN          0
//minimum negative value for an activation variable
#define SPINN_ACTIV_MIN_NEG      (-1 * SPINN_ACTIV_MAX)
#define SPINN_ACTIV_NaN          (-1 << SPINN_ACTIV_SHIFT)
// long activations are 32-bit quantities with 15 decimal bits
#define SPINN_LONG_ACTIV_MAX     INT_MAX
#define SPINN_LONG_ACTIV_MIN     0
//minimum negative value for a long activation variable
#define SPINN_LONG_ACTIV_MIN_NEG INT_MIN
//long long activations are 64-bit quantities with 15 decimal bits
#define SPINN_LLONG_ACTIV_MAX    LLONG_MAX
#define SPINN_LLONG_ACTIV_MIN    0
//minimum negative value for a long long activation variable
#define SPINN_LLONG_ACTIV_MIN_NEG  LLONG_MIN
//these values are set to compute the cross entropy error function
#define SPINN_LONG_ACTIV_ONE     (1 << SPINN_ACTIV_SHIFT)
#define SPINN_LONG_ACTIV_NEG_ONE (-1 << SPINN_ACTIV_SHIFT)


typedef short     derivative_t;     // input or output derivative (16 bit)
typedef int       long_deriv_t;     // intermediate unit input or output derivative (32 bit)
typedef long long llong_deriv_t;    // intermediate unit input or output derivative (64 bit)

// derivatives are 16-bit quantities with 15 decimal bits
#define SPINN_DERIV_SHIFT        SPINN_ACTIV_SHIFT
#define SPINN_DERIV_MAX          SPINN_ACTIV_MAX
#define SPINN_DERIV_MIN          SPINN_ACTIV_MIN
//minimum negative value for an derivative variable
#define SPINN_DERIV_MIN_NEG      SPINN_ACTIV_MIN_NEG
#define SPINN_DERIV_NaN          SPINN_ACTIV_NaN
// long derivatives are 32-bit quantities with 15 decimal bits
#define SPINN_LONG_DERIV_MAX     SPINN_LONG_ACTIV_MAX
#define SPINN_LONG_DERIV_MIN     SPINN_LONG_ACTIV_MIN
//minimum negative value for a long derivative variable
#define SPINN_LONG_DERIV_MIN_NEG SPINN_LONG_ACTIV_MIN_NEG
//long long derivative are 64-bit quantities with 15 decimal bits
#define SPINN_LLONG_DERIV_MAX    SPINN_LLONG_ACTIV_MAX
#define SPINN_LLONG_DERIV_MIN    SPINN_LLONG_ACTIV_MIN
//minimum negative value for a long long derivative variable
#define SPINN_LLONG_DERIV_MIN_NEG  SPINN_LLONG_ACTIV_MIN_NEG
//these values are set to compute the cross entropy error function
#define SPINN_LONG_DERIV_ONE     SPINN_LONG_ACTIV_ONE
#define SPINN_LONG_DERIV_NEG_ONE SPINN_LONG_ACTIV_NEG_ONE

typedef int       net_t;            // unit internal net (inputs ot-product)
typedef long long long_net_t;       // used for net intermediate calc

//TODO: set these values correctly!
// nets are 32-bit quantities with 27 decimal bits
#define SPINN_NET_SHIFT          (SPINN_WEIGHT_SHIFT + SPINN_ACTIV_SHIFT)
#define SPINN_NET_MAX            ( 15.0 * (1 << SPINN_NET_SHIFT))
#define SPINN_NET_MIN            (-15.0 * (1 << SPINN_NET_SHIFT))

typedef int       error_t;          // unit output error
typedef long long long_error_t;     // used for error intermediate calc
typedef int       delta_t;          // input derivative
typedef long long long_delta_t;     // used for delta intermediate calc

//TODO: set these values correctly!
// errors are 32-bit quantities with 15 decimal bits
#define SPINN_ERROR_SHIFT        SPINN_ACTIV_SHIFT
#define SPINN_ERROR_MAX          ( 0.5 * (1 << SPINN_ERROR_SHIFT))
#define SPINN_ERROR_MIN          (-0.5 * (1 << SPINN_ERROR_SHIFT))
// intermediate error computations use longer types!  (64-bit quantities with 27 decimal bits)
#define SPINN_LONG_ERR_SHIFT     (SPINN_WEIGHT_SHIFT + SPINN_ERROR_SHIFT)
#define SPINN_LONG_ERR_MAX       ( 0.5 * (1 << SPINN_LONG_ERR_SHIFT))
#define SPINN_LONG_ERR_MIN       (-0.5 * (1 << SPINN_LONG_ERR_SHIFT))
//deltas are 32-bit quantities with 15 decimal bits
#define SPINN_DELTA_SHIFT        SPINN_ERROR_SHIFT
#define SPINN_DELTA_MAX          SPINN_ERROR_MAX
#define SPINN_DELTA_MIN          SPINN_ERROR_MIN
// intermediate delta computations use longer types!  (64-bit quantities with 27 decimal bits)
#define SPINN_LONG_DELTA_SHIFT   SPINN_LONG_ERR_SHIFT
#define SPINN_LONG_DELTA_MAX     SPINN_LONG_ERR_MAX
#define SPINN_LONG_DELTA_MIN     SPINN_LONG_ERR_MIN

typedef short     weight_t;         // connection weight
typedef int       wchange_t;        // accumulated connection weight change

// weights are 16-bit quantities with 12 decimal bits
#define SPINN_WEIGHT_SHIFT       12
#define SPINN_WEIGHT_MAX         ((weight_t) ( 7 << SPINN_WEIGHT_SHIFT))
#define SPINN_WEIGHT_MIN         ((weight_t) (-7 << SPINN_WEIGHT_SHIFT))
#define SPINN_WEIGHT_POS_DELTA   ((weight_t)  1)
#define SPINN_WEIGHT_NEG_DELTA   ((weight_t) -1)

typedef uint      scoreboard_t;     // keep track of received items

typedef uchar     proc_phase_t;     // phase (FORWARD or BACKPROP)

typedef int       fpreal;           // int as 16.16 fixed-point number
typedef long long lfpreal;          // int as 48.16 fixed-point number

#define SPINN_FPREAL_SHIFT       16
//#define SPINN_FP_NaN             0xffff0000
#define SPINN_FP_NaN             (-1 << SPINN_FPREAL_SHIFT)
#define SPINN_SMALL_VAL          1


// ------------------------------------------------------------------------
// global (network-wide) configuration
// ------------------------------------------------------------------------
typedef struct global_conf      // MLP configuration
{
  // neural net configuration parameters
  uchar net_type;               // type of neural net
  uchar training;               // training or testing mode?
  uint  num_epochs;             // number of epochs to run
  uint  num_examples;           // number of examples per epoch
  uint  ticks_per_int;          // number of ticks per interval
  uint  max_unit_outs;          // max. number of outputs in a group
  uint  num_chips;              // number of chips in the simulation
  uint  timeout;                // in case something goes wrong
  uint  chip_struct_addr;       // address in SDRAM for core map
  uint  global_max_ticks;       // max number of ticks across all the examples
} global_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// chip-wide configuration
// ------------------------------------------------------------------------
typedef struct chip_struct      // chip specific information
{
  // SpiNNaker configuration parameters
  uchar core_type[SPINN_NUM_CORES_CHIP];
  uint  conf_size;              // largest configuration data size
  uint  num_rt_entries;         // number of routing table entries
  uint  num_write_blks;         // number of groups that write outputs
  uint  cm_struct_addr;         // address in SDRAM for core map
  uint  core_struct_addr[SPINN_NUM_CORES_CHIP]; // addr in SDRAM for core data
  uint  rt_struct_addr;         // address in SDRAM for router data
  uint  example_set_addr;       // address in SDRAM for example set file
  uint  examples_addr;          // address in SDRAM for examples file
  uint  events_addr;            // address in SDRAM for events file
} chip_struct_t;
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
  activation_t learningRate;        // network learning rate
  uint         weights_struct_addr; // address in SDRAM for weight file
} w_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// sum core configuration
// ------------------------------------------------------------------------
// sum cores accumulate acummulate b-d-ps sent by weight cores and
// compute unit nets (FORWARD phase) and errors (BACKPROP phase)
// ------------------------------------------------------------------------
typedef struct s_conf    // sum core configuration
{
  uchar        output_grp;          // is this an OUTPUT group?
  uchar        input_grp;           // is this an INPUT group?
  uint         num_nets;            // this core's number of unit nets
  uint         net_blk;             // this core's net block
  uint         error_blk;           // this core's error block
  scoreboard_t f_all_arrived;       // all expected unit output b-d-ps
  scoreboard_t f_all_done;          // all unit outputs
  scoreboard_t b_all_arrived;       // all expected error d-b-ps
  scoreboard_t b_all_done;          // all error deltas
  uint         num_in_procs;        // number of input (net) comp procedures
  uint         procs_list[SPINN_NUM_IN_PROCS];
  uchar        in_integr_en;        // input integrator in use
  fpreal       in_integr_dt;        // integration time const for input integr
  fpreal       soft_clamp_strength; // Strength coeff for soft clamp fix 16.16
  net_t        initNets;            // initial value for unit nets
  activation_t initOutput;          // initial value for unit outputs
  uint         inputs_addr;         // address in SDRAM for inputs file
} s_conf_t;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// input core configuration
// ------------------------------------------------------------------------
// input cores process the values received from the sum cores through the
// elements of the input pipeline as required by LENS.
// ------------------------------------------------------------------------
typedef struct i_conf               // sum core configuration
{
  uchar        output_grp;          // is this an OUTPUT group?
  uchar        input_grp;           // is this an INPUT group?
  uint         num_nets;            // this core's number of unit nets
  uint         net_blk;             // this core's net block
  uint         delta_blk;           // this core's delta block
  scoreboard_t f_all_arrived;       // all expected unit output b-d-ps
  scoreboard_t f_all_done;          // all unit outputs
  scoreboard_t b_all_arrived;       // all expected error d-b-ps
  scoreboard_t b_all_done;          // all error deltas
  uint         num_in_procs;        // number of input (net) comp procedures
  uint         procs_list[SPINN_NUM_IN_PROCS];
  uchar        in_integr_en;        // input integrator in use
  fpreal       in_integr_dt;        // integration time const for input integr
  fpreal       soft_clamp_strength; // Strength coeff for soft clamp fix 16.16
  net_t        initNets;            // initial value for unit nets
  activation_t initOutput;          // initial value for unit outputs
  uint         inputs_addr;         // address in SDRAM for inputs file
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
  activation_t initOutput;            // initial value for unit outputs
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


typedef struct mc_table_entry // multicast routing entry
{
  uint key;               // entry key
  uint mask;              // entry mask
  uint route;             // entry route
} mc_table_entry_t;


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
// Examples are organized as LENS exmaples (see LENS documentation).
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
