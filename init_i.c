// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "sdram.h"

#include "comms_i.h"

// this files contains the initialization routine for I cores

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint coreID;               // 5-bit virtual core ID
extern uint coreIndex;            // coreID - 1 (convenient for array indexing)
extern uint fwdKey;               // 32-bit packet ID for FORWARD phase
extern uint bkpKey;               // 32-bit packet ID for BACKPROP phase
extern uint stpKey;               // 32-bit packet ID for stop criterion

extern uint coreType;             // weight, sum or threshold

extern uint         example;      // current example in epoch
extern uint         num_events;   // number of events in current example
extern uint         event_idx;    // index into current event
extern uint         num_ticks;    // number of ticks in current event
extern uint         max_ticks;    // maximum number of ticks in current event
extern uint         min_ticks;    // minimum number of ticks in current event
extern uint         tick;         // current tick in phase

extern chip_struct_t        *ct; // chip-specific data
extern uint                 *cm; // simulation core map
extern uchar                *dt; // core-specific data
extern mc_table_entry_t     *rt; // multicast routing table data
extern weight_t             *wt; // initial connection weights
extern struct mlp_set       *es; // example set data
extern struct mlp_example   *ex; // example data
extern struct mlp_event     *ev; // event data
extern activation_t         *it; // example inputs
extern activation_t         *tt; // example targets

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern global_conf_t  mlpc;       // network-wide configuration parameters
extern chip_struct_t  ccfg;       // chip configuration parameters
extern i_conf_t       icfg;       // input core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// input core variables
// ------------------------------------------------------------------------
extern long_net_t     * i_nets;        // unit nets computed in current tick
extern long_error_t   * i_errors;      // errors computed in current tick
extern long_error_t   * i_init_err;    // errors computed in first tick
extern pkt_queue_t      i_pkt_queue;   // queue to hold received b-d-ps
extern uchar            i_active;      // processing b-d-ps from queue?
extern uint             i_it_idx;      // index into current inputs/targets
extern scoreboard_t   * if_arrived;    // keep track of expected net b-d-p
extern scoreboard_t     if_done;       // current tick net computation done
extern uint             if_thrds_done; // sync. semaphore: proc & stop
extern long_error_t   * ib_init_error; // initial error value for every tick
extern scoreboard_t     ib_all_arrived;// all deltas have arrived in tick
extern scoreboard_t   * ib_arrived;    // keep track of expected error b-d-p
extern scoreboard_t     ib_done;       // current tick error computation done
//#extern uint             ib_thrds_done; // sync. semaphore: proc & stop
extern long_net_t     * i_last_integr_output;   //last integrator output value
//list of input pipeline procedures
extern in_proc_t const  i_in_procs[SPINN_NUM_IN_PROCS];
//list of initialization procedures for input pipeline
extern in_proc_init_t const  i_init_in_procs[SPINN_NUM_IN_PROCS];
extern long_net_t      * i_input_history; //sdram pointer where to store input history
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
#ifdef DEBUG
  extern uint pkt_sent;  // total packets sent
  extern uint sent_fwd;  // packets sent in FORWARD phase
#endif
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory and initialize variables
// ------------------------------------------------------------------------
uint i_init (void)
{
  uint i;

  // allocate memory for nets
  if ((i_nets = ((long_net_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((i_errors = ((long_error_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // TODO: probably this variable can be removed
  // allocate memory to store error values during the first BACKPROPagation tick
  if ((i_init_err = ((long_error_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((i_pkt_queue.queue = ((packet_t *)
         spin1_malloc (SPINN_INPUT_PQ_LEN * sizeof(packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received net b-d-ps scoreboards
  if ((if_arrived = ((scoreboard_t *)
          spin1_malloc (icfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received error b-d-ps scoreboards
  if ((ib_arrived = ((scoreboard_t *)
          spin1_malloc (icfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }
  
  // intialize tick
  //NOTE: input cores do not have a tick 0
  tick = SPINN_I_INIT_TICK;

  // initialize nets, errors and scoreboards
  for (i = 0; i < icfg.num_nets; i++)
  {
    i_nets[i] = 0;
    i_errors[i] = 0;
    if_arrived[i] = 0;
    ib_arrived[i] = 0;
  }
  if_done = 0;
  ib_done = 0;

  #if SPLIT_ARR == TRUE
    ib_all_arrived = icfg.b_init_arrived + icfg.b_next_arrived; //#
  #else
    ib_all_arrived = icfg.b_all_arrived;
  #endif

  // initialize synchronization semaphores
  if_thrds_done = 1;

  // initialize processing thread flag
  i_active = FALSE;

  // initialize packet queue
  i_pkt_queue.head = 0;
  i_pkt_queue.tail = 0;

  // initialize packet keys
  //NOTE: colour is initialized to 0.
  fwdKey = SPINN_SB_KEY(icfg.net_blk)   | SPINN_CORETYPE_KEY
             | SPINN_PHASE_KEY(SPINN_FORWARD);
  bkpKey = SPINN_SB_KEY(icfg.error_blk) | SPINN_CORETYPE_KEY
             | SPINN_PHASE_KEY(SPINN_BACKPROP);

  // if input or output group initialize event input/target index
  if (icfg.input_grp || icfg.output_grp)
  {
    i_it_idx = ev[event_idx].it_idx * icfg.num_nets;
  }
  
  // if the network requires training and elements of the pipeline require
  // initialization, then follow the appropriate procedure
  // use the list of procedures in use from lens and call the appropriate
  // initialization routine from the i_init_in_procs function pointer list
  
  for (i = 0; i < icfg.num_in_procs; i++)
    if (i_init_in_procs[icfg.procs_list[i]] != NULL)
    {
      int return_value;
      // call the appropriate routine for pipeline initialization
      return_value = i_init_in_procs[icfg.procs_list[i]]();
      
      // if return value contains error, return it
      if (return_value != SPINN_NO_ERROR)
          return return_value;
    }
/*
  // allocate memory in SDRAM for input history
  // TODO: this need a condition on the requirement to have input history
  // this needs to come from splens
  if ((i_input_history = ((long_net_t *)
          sark_xalloc (sv->sdram_heap,
                       icfg.num_nets * mlpc.global_max_ticks * sizeof(long_net_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }
*/

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
