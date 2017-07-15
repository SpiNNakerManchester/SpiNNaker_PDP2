// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"

#include "comms_s.h"

// this files contains the initialization routine for S cores

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint coreID;               // 5-bit virtual core ID
extern uint coreIndex;            // coreID - 1 (convenient for array indexing)

extern uint coreType;             // weight, sum, input or threshold

extern uint fwdKey;               // 32-bit packet ID for FORWARD phase
extern uint bkpKey;               // 32-bit packet ID for BACKPROP phase

extern uint         example;      // current example in epoch
extern uint         num_events;   // number of events in current example
extern uint         event_idx;    // index into current event
extern uint         num_ticks;    // number of ticks in current event
extern uint         max_ticks;    // maximum number of ticks in current event
extern uint         min_ticks;    // minimum number of ticks in current event
extern uint         tick;         // current tick in phase

extern chip_struct_t        *ct; // chip-specific data
extern uchar                *dt; // core-specific data
//extern mc_table_entry_t     *rt; // multicast routing table data
extern uint                 *rt; // multicast routing keys data
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
extern s_conf_t       scfg;       // sum core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// sum core variables
// ------------------------------------------------------------------------
extern long_net_t     * s_nets[2];     // unit nets computed in current tick
extern long_error_t   * s_errors[2];   // errors computed in current tick
extern long_error_t   * s_init_err[2]; // errors computed in first tick
extern pkt_queue_t      s_pkt_queue;   // queue to hold received b-d-ps
extern uchar            s_active;      // processing b-d-ps from queue?
extern scoreboard_t   * sf_arrived[2]; // keep track of expected net b-d-p
extern scoreboard_t     sf_done;       // current tick net computation done
extern uint             sf_thrds_done; // sync. semaphore: proc & stop
extern long_error_t   * sb_init_error; // initial error value for every tick
extern scoreboard_t     sb_all_arrived;// all deltas have arrived in tick
extern scoreboard_t   * sb_arrived[2]; // keep track of expected error b-d-p
extern scoreboard_t     sb_done;       // current tick error computation done
//#extern uint             sb_thrds_done; // sync. semaphore: proc & stop
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
uint s_init (void)
{
  uint i;

  // allocate memory for nets
  if ((s_nets[0] = ((long_net_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_nets[1] = ((long_net_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((s_errors[0] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_errors[1] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for the first tick of errors in BACKPROP phase
  //TODO: is this necessary? -- not used anywhere
  if ((s_init_err[0] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  //TODO: is this necessary? -- not used anywhere
  if ((s_init_err[1] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((s_pkt_queue.queue = ((packet_t *)
         spin1_malloc (SPINN_SUM_PQ_LEN * sizeof(packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received net b-d-ps scoreboards
  if ((sf_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sf_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received error b-d-ps scoreboards
  if ((sb_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sb_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // intialize tick
  //NOTE: SUM cores do not have a tick 0
  tick = SPINN_S_INIT_TICK;

  // initialize nets, errors and scoreboards
  for (i = 0; i < scfg.num_nets; i++)
  {
    s_nets[0][i] = 0;
    s_nets[1][i] = 0;
    s_errors[0][i] = 0;
    s_errors[1][i] = 0;
    sf_arrived[0][i] = 0;
    sf_arrived[1][i] = 0;
    sb_arrived[0][i] = 0;
    sb_arrived[1][i] = 0;
  }
  sf_done = 0;
  sb_done = 0;

  #if SPLIT_ARR == TRUE
    sb_all_arrived = scfg.b_init_arrived + scfg.b_next_arrived; //#
  #else
    sb_all_arrived = scfg.b_all_arrived;
  #endif

  // initialize synchronization semaphores
  sf_thrds_done = 1;

  // initialize processing thread flag
  s_active = FALSE;

  // initialize packet queue
  s_pkt_queue.head = 0;
  s_pkt_queue.tail = 0;

  // initialize packet keys
  //NOTE: colour is initialized to 0.
//lap  fwdKey = SPINN_SB_KEY(scfg.net_blk)   | SPINN_CORETYPE_KEY
//lap             | SPINN_PHASE_KEY(SPINN_FORWARD);
//lap  bkpKey = SPINN_SB_KEY(scfg.error_blk) | SPINN_CORETYPE_KEY
//lap             | SPINN_PHASE_KEY(SPINN_BACKPROP);
  fwdKey = rt[FWD] | SPINN_CORETYPE_KEY
             | SPINN_PHASE_KEY(SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_CORETYPE_KEY
             | SPINN_PHASE_KEY(SPINN_BACKPROP);

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
