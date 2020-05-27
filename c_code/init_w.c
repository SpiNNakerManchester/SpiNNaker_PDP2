// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_w.h"

// this file contains the initialisation routine for W cores

// ------------------------------------------------------------------------
// allocate memory and initialise variables
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
  if ((w_errors = ((error_t *)
         spin1_malloc (wcfg.num_rows * sizeof(error_t)))) == NULL
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



  // initialise weights from SDRAM
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

  // initialise link deltas
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    for (uint j = 0; j < wcfg.num_cols; j++)
    {
      w_link_deltas[i][j] = 0;
    }
  }

  // initialise weight changes
  for (i = 0; i < wcfg.num_rows; i++)
  {
    for (j = 0; j < wcfg.num_cols; j++)
    {
      w_wchanges[i][j] = 0;
    }
  }

  // initialise error dot products
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    w_errors[i] = 0;
  }

  // initialise output history for tick 0
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    w_output_history[i] = 0;
  }

  // initialise tick
  tick = SPINN_W_INIT_TICK;

  // initialise delta scaling factor
  // s15.16
  w_delta_dt = (1 << SPINN_FPREAL_SHIFT) / ncfg.ticks_per_int;

  // initialise pointers to received unit outputs
  wf_procs = 0;
  wf_comms = 1;

  // initialise synchronisation semaphores
  wf_thrds_pend = 0; // just wait for initial unit outputs
  wb_thrds_pend = 0; // just wait for initial deltas

  // initialise processing thread flag
  wb_active = FALSE;

  // initialise arrival scoreboards
  wf_arrived = 0;
  wb_arrived = 0;


  // set weight update function
  wb_update_func = w_update_procs[wcfg.update_function];

  // initialise packet keys
  //NOTE: colour is initialised to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY(SPINN_FORWARD)
		  | SPINN_BLOCK_KEY(wcfg.col_blk);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY(SPINN_BACKPROP)
		  | SPINN_BLOCK_KEY(wcfg.row_blk);

  wf_sync_key = rt[FDS] | SPINN_SYNC_KEY | SPINN_PHASE_KEY(SPINN_FORWARD);
  wb_sync_key = rt[FDS] | SPINN_SYNC_KEY | SPINN_PHASE_KEY(SPINN_BACKPROP);

  ldsaKey = rt[LDS] | SPINN_LDSA_KEY;

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stage start callback: get stage started
// ------------------------------------------------------------------------
void stage_start (void)
{
  // start log
  io_printf (IO_BUF, "--------------\n");
  io_printf (IO_BUF, "starting stage\n");
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// check exit code and print details of the state
// ------------------------------------------------------------------------
void stage_done (uint ec)
{
  // pause timer and setup next stage,
  simulation_handle_pause_resume (NULL);

  // report problems -- if any
  switch (ec)
  {
    case SPINN_NO_ERROR:
      io_printf (IO_BUF, "stage OK\n");
      break;

    case SPINN_CFG_UNAVAIL:
      io_printf (IO_BUF, "core configuration failed\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_QUEUE_FULL:
      io_printf (IO_BUF, "packet queue full\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_MEM_UNAVAIL:
      io_printf (IO_BUF, "malloc failed\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_UNXPD_PKT:
      io_printf (IO_BUF, "unexpected packet received - abort!\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_TIMEOUT_EXIT:
      io_printf (IO_BUF, "timeout (h:%u e:%u p:%u t:%u) - abort!\n",
                 epoch, example, phase, tick
                );
      io_printf (IO_BUF, "stage aborted\n");
#ifdef DEBUG_TO
      io_printf (IO_BUF, "(fp:%u  fc:%u)\n", wf_procs, wf_comms);
      io_printf (IO_BUF, "(fptd:%u)\n", wf_thrds_pend);
      io_printf (IO_BUF, "(fa:%u ba:%u)\n",
                 wf_arrived, wb_arrived
                );
#endif
      break;
  }

#ifdef DEBUG
  // report diagnostics
  io_printf (IO_BUF, "total ticks:%d\n", tot_tick);
  io_printf (IO_BUF, "total recv:%d\n", pkt_recv);
  io_printf (IO_BUF, "total sent:%d\n", pkt_sent);
  io_printf (IO_BUF, "recv: fwd:%d bkp:%d\n", recv_fwd, recv_bkp);
  io_printf (IO_BUF, "sent: fwd:%d bkp:%d\n", sent_fwd, sent_bkp);
  io_printf (IO_BUF, "unused recv: fwd:%d bkp:%d\n", pkt_fwbk, pkt_bwbk);
  io_printf (IO_BUF, "sync sent:%d\n", spk_sent);
  io_printf (IO_BUF, "ldsa sent:%d\n", lda_sent);
  io_printf (IO_BUF, "ldsr recv:%d\n", ldr_recv);
  io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
  io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
  if (wrng_phs) io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
  if (wrng_tck) io_printf (IO_BUF, "wrong tick:%d\n", wrng_tck);
  if (wrng_btk) io_printf (IO_BUF, "wrong btick:%d\n", wrng_btk);
  io_printf (IO_BUF, "------\n");
  io_printf (IO_BUF, "weight updates:%d\n", wght_ups);
#endif

  // close log,
  io_printf (IO_BUF, "stopping stage\n");
  io_printf (IO_BUF, "--------------\n");

  // and let host know that we're done
  if (ec == SPINN_NO_ERROR) {
    simulation_ready_to_read ();
  } else {
    rt_error (RTE_SWERR);
  }
}
// ------------------------------------------------------------------------
