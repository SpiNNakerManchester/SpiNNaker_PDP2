// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_w.h"

// this files contains the initialization routine for W cores

// ------------------------------------------------------------------------
// allocate memory and initialize variables
// ------------------------------------------------------------------------
uint w_init (void)
{
  uint i, j;

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "r:%d c:%d\n",
                wcfg.num_rows,
                wcfg.num_cols
              );
  #endif

  // TODO: the following memory allocation is to be used to store
  // the history of any of these sets of values. When training
  // continuous networks, these histories always need to be saved.
  // For non-continuous networks, they only need to be stored if the
  // backpropTicks field of the network is greater than one. This
  // information needs to come in the wcfg structure.

  // allocate memory in SDRAM for output history
  if ((w_output_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       wcfg.num_rows * ncfg.global_max_ticks * sizeof(activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for weights
  if ((w_weights = ((weight_t * *)
         spin1_malloc (wcfg.num_rows * sizeof(weight_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_weights[i] = ((weight_t *)
           spin1_malloc (wcfg.num_cols * sizeof(weight_t)))) == NULL
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
  if ((w_outputs[0] = ((activation_t *)
         spin1_malloc (wcfg.num_rows * sizeof(activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((w_outputs[1] = ((activation_t *)
         spin1_malloc (wcfg.num_rows * sizeof(activation_t)))) == NULL
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
  //NOTE: could use DMA
  for (i = 0; i < wcfg.num_rows; i++)
  {
    spin1_memcpy (w_weights[i],
                   &wt[i * wcfg.num_cols],
                   wcfg.num_cols * sizeof(weight_t)
                 );
  }

  #ifdef DEBUG_CFG2
    for (uint r = 0; r < wcfg.num_rows; r++)
    {
      for (uint c =0; c < wcfg.num_cols; c++)
      {
	    io_printf (IO_BUF, "w[%u][%u]: %k\n", r, c, w_weights[r][c]);
      }
    }
  #endif

  #ifdef WEIGHT_TRACE
    //TODO: dump weights to SDRAM for record keeping
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

  // initialize tick
  tick = SPINN_W_INIT_TICK;

  // initialize delta scaling factor
  // s15.16
  w_delta_dt = (1 << SPINN_FPREAL_SHIFT) / ncfg.ticks_per_int;

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


  // set weight update function
  wb_update_func = w_update_procs[wcfg.update_function];

  // initialize packet keys
  //NOTE: colour is initialized to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY(SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY(SPINN_BACKPROP);

  wf_sync_key = rt[FDS] | SPINN_SYNC_KEY | SPINN_PHASE_KEY(SPINN_FORWARD);
  wb_sync_key = rt[FDS] | SPINN_SYNC_KEY | SPINN_PHASE_KEY(SPINN_BACKPROP);

  ldsKey = rt[LDS];

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
