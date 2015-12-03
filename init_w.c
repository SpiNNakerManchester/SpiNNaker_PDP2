// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "sdram.h"

#include "comms_w.h"

// this files contains the initialization routine for W cores

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
extern short_weight_t       *wt; // initial connection weights
extern struct mlp_set       *es; // example set data
extern struct mlp_example   *ex; // example data
extern struct mlp_event     *ev; // event data
extern short_activ_t        *it; // example inputs
extern short_activ_t        *tt; // example targets

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern global_conf_t  mlpc;       // network-wide configuration parameters
extern chip_struct_t  ccfg;       // chip configuration parameters
extern w_conf_t       wcfg;       // weight core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// weight core variables
// ------------------------------------------------------------------------
extern short_weight_t * * w_weights;     // connection weights block
extern long_wchange_t * * w_wchanges;    // accumulated weight changes
extern short_activ_t  * w_outputs[2]; // unit outputs for b-d-p
extern long_delta_t * * w_link_deltas; // computed link deltas
extern error_t        * w_errors;      // computed errors next tick
extern pkt_queue_t      w_delta_pkt_q; // queue to hold received deltas
extern fpreal           w_delta_dt;    // scaling factor for link deltas
extern uint             wf_procs;      // pointer to processing unit outputs
extern uint             wf_comms;      // pointer to receiving unit outputs
extern scoreboard_t     wf_arrived;    // keeps track of received unit outputs
extern uint             wf_thrds_done; // sync. semaphore: comms, proc & stop
extern uint             wf_sync_key;   // FORWARD processing can start
extern uchar            wb_active;     // processing deltas from queue?
extern scoreboard_t     wb_arrived;    // keeps track of received deltas
extern uint             wb_sync_key;   // BACKPROP processing can start
// history arrays
extern short_activ_t  * w_output_history;
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
uint w_init (void)
{
  uint i, j;

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "br:%d bc:%d r:%d c:%d\n",
                wcfg.blk_row,
                wcfg.blk_col,
                wcfg.num_rows,
                wcfg.num_cols
              );
  #endif

  // TODO: the following memory allocation is to be used to store
  // the history of any of these sets of values. When training
  // continuous networks, these histories always need to be saved.
  // For non-continuous networks, they only need to be stored if the 
  // backpropTicks field of the network is greater than one. This
  // information needs to come from splens in the tcfg structure.

  // allocate memory in SDRAM for output history
  if ((w_output_history = ((short_activ_t *)
          sark_xalloc (sv->sdram_heap,
                       wcfg.num_rows * mlpc.global_max_ticks * sizeof(short_activ_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for weights
  if ((w_weights = ((short_weight_t * *)
         spin1_malloc (wcfg.num_rows * sizeof(short_weight_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_weights[i] = ((short_weight_t *)
           spin1_malloc (wcfg.num_cols * sizeof(short_weight_t)))) == NULL
       )
    {
    return (SPINN_MEM_UNAVAIL);
    }
  }

  // allocate memory for weight changes
  if ((w_wchanges = ((long_wchange_t * *)
         spin1_malloc (wcfg.num_rows * sizeof(long_wchange_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_wchanges[i] = ((long_wchange_t *)
           spin1_malloc (wcfg.num_cols * sizeof(long_wchange_t)))) == NULL
       )
    {
    return (SPINN_MEM_UNAVAIL);
    }
  }

  // allocate memory for unit outputs
  if ((w_outputs[0] = ((short_weight_t *)
         spin1_malloc (wcfg.num_rows * sizeof(short_activ_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((w_outputs[1] = ((short_activ_t *)
         spin1_malloc (wcfg.num_rows * sizeof(short_activ_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for link deltas
  if ((w_link_deltas = ((long_delta_t * *)
         spin1_malloc (wcfg.num_rows * sizeof(long_delta_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_link_deltas[i] = ((long_delta_t *)
           spin1_malloc (wcfg.num_cols * sizeof(long_delta_t)))) == NULL
       )
    {
    return (SPINN_MEM_UNAVAIL);
    }
  }

  // allocate memory for errors
  if ((w_errors = ((error_t*)
         spin1_malloc (wcfg.num_rows * sizeof(delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((w_delta_pkt_q.queue = ((packet_t *)
         spin1_malloc (SPINN_WEIGHT_PQ_LEN * sizeof(packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }



  // initialize weights from SDRAM
  wt = (short_weight_t *) wcfg.weights_struct_addr;  // initial connection weights

  //NOTE: could use DMA
  for (i = 0; i < wcfg.num_rows; i++)
  {
    spin1_memcpy (w_weights[i],
                   &wt[i * wcfg.num_cols],
                   wcfg.num_cols * sizeof(short_weight_t)
                 );
  }

  #ifdef WEIGHT_TRACE
    // dump weights to SDRAM for record keeping
    //NOTE: could use DMA
    //TODO: need to recompute indices!
    for (i = 0; i < wcfg.num_rows; i++)
    {
      spin1_memcpy (&wh[((wcfg.blk_row * wcfg.num_rows + i) * mlpc.num_outs)
                     + (wcfg.blk_col * wcfg.num_cols)],
                     w_weights[i],
                     wcfg.num_cols * sizeof(short_weight_t)
                   );
    }
  #endif

  // initialize link deltas
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    for (uint j = 0; j < wcfg.num_cols; j++)
    {
      w_link_deltas[i][j] = 0;
    }
  }

  // initialize weight changes
  for (i = 0; i < wcfg.num_rows; i++)
  {
    for (j = 0; j < wcfg.num_cols; j++)
    {
      w_wchanges[i][j] = 0;
    }
  }

  // initialize error dot products
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    w_errors[i] = 0;
  }

  // initialize output history for tick 0
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    w_output_history[i] = 0;
  }

  // intialize tick
  tick = SPINN_W_INIT_TICK;

  // intialize delta scaling factor
  // s15.16
  w_delta_dt = (1 << SPINN_FPREAL_SHIFT) / mlpc.ticks_per_int;

  // initialize pointers to received unit outputs
  wf_procs = 0;
  wf_comms = 1;

  // initialize synchronization semaphores
  wf_thrds_done = 0; // just wait for initial unit outputs

  // initialize processing thread flag
  wb_active = FALSE;

  // initialize arrival scoreboards
  wf_arrived = 0;
  wb_arrived = 0;

  // initialize packet keys
  //NOTE: colour is initialized to 0.
  uint block_key = SPINN_BR_KEY(wcfg.blk_row) | SPINN_BC_KEY(wcfg.blk_col);
  uint base_key = block_key | SPINN_CORETYPE_KEY;

  fwdKey = base_key | SPINN_PHASE_KEY(SPINN_FORWARD);
  bkpKey = base_key | SPINN_PHASE_KEY(SPINN_BACKPROP);

  wf_sync_key = block_key | SPINN_SYNC_KEY | SPINN_PHASE_KEY(SPINN_FORWARD);
  wb_sync_key = block_key | SPINN_SYNC_KEY | SPINN_PHASE_KEY(SPINN_BACKPROP);

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
