// SpiNNaker API
#include "spin1_api.h"

// front-end-common
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"
#include "init_w.h"
#include "comms_w.h"
#include "process_w.h"


// ------------------------------------------------------------------------
// weight core initialisation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// load configurations from SDRAM
// ------------------------------------------------------------------------
uint cfg_init (void)
{
#ifdef DEBUG
  io_printf (IO_BUF, "weight\n");
#endif

  // read the data specification header
  data_specification_metadata_t * data =
          data_specification_get_data_address();
  if (!data_specification_read_header (data))
  {
    return (SPINN_CFG_UNAVAIL);
  }

  // set up the simulation interface (system region)
  //NOTE: these variables are not used!
  uint32_t run_forever;
  if (!simulation_steps_initialise(
      data_specification_get_region(SYSTEM, data),
      APPLICATION_NAME_HASH, &stage_num_steps,
      &run_forever, &stage_step, 0, 0)
      )
  {
    return (SPINN_CFG_UNAVAIL);
  }

  // network configuration address
  address_t nt = data_specification_get_region (NETWORK, data);

  // initialise network configuration from SDRAM
  spin1_memcpy (&ncfg, nt, sizeof (network_conf_t));

  // core configuration address
  address_t dt = data_specification_get_region (CORE, data);

  // initialise core-specific configuration from SDRAM
  spin1_memcpy (&wcfg, dt, sizeof (w_conf_t));

  // initial connection weights
  wt = (weight_t *) data_specification_get_region
      (WEIGHTS, data);

  // example set
  es = (mlp_set_t *) data_specification_get_region
      (EXAMPLE_SET, data);

  // examples
  ex = (mlp_example_t *) data_specification_get_region
      (EXAMPLES, data);

  // routing keys
  rt = (uint *) data_specification_get_region
      (ROUTING, data);

  // initialise stage configuration from SDRAM
  xadr = data_specification_get_region (STAGE, data);
  spin1_memcpy (&xcfg, xadr, sizeof (stage_conf_t));

#ifdef DEBUG
  io_printf (IO_BUF, "stage %u configured\n", xcfg.stage_id);
  if (xcfg.training)
  {
    io_printf (IO_BUF, "train (updates:%u)\n", xcfg.num_epochs);
  }
  else
  {
    io_printf (IO_BUF, "test (examples:%u)\n", xcfg.num_examples);
  }
#endif

#ifdef DEBUG_CFG
  io_printf (IO_BUF, "nr: %d\n", wcfg.num_rows);
  io_printf (IO_BUF, "nc: %d\n", wcfg.num_cols);
  io_printf (IO_BUF, "rb: %d\n", wcfg.row_blk);
  io_printf (IO_BUF, "cb: %d\n", wcfg.col_blk);
  io_printf (IO_BUF, "lr: %k\n", wcfg.learningRate);
  io_printf (IO_BUF, "wd: %k\n", wcfg.weightDecay);
  io_printf (IO_BUF, "mm: %k\n", wcfg.momentum);
  io_printf (IO_BUF, "uf: %d\n", xcfg.update_function);
  io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
  io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
  io_printf (IO_BUF, "sk: 0x%08x\n", rt[FDS]);
  io_printf (IO_BUF, "ld: 0x%08x\n", rt[LDS]);
#endif

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory in DTCM and SDRAM
// ------------------------------------------------------------------------
uint mem_init (void)
{
  // allocate memory for weights
  if ((w_weights = ((weight_t * *)
         spin1_malloc (wcfg.num_rows * sizeof (weight_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_weights[i] = ((weight_t *)
         spin1_malloc (wcfg.num_cols * sizeof (weight_t)))) == NULL
       )
    {
    return (SPINN_MEM_UNAVAIL);
    }
  }

  // allocate memory for weight changes
  if ((w_wchanges = ((long_wchange_t * *)
         spin1_malloc (wcfg.num_rows * sizeof (long_wchange_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_wchanges[i] = ((long_wchange_t *)
         spin1_malloc (wcfg.num_cols * sizeof (long_wchange_t)))) == NULL
       )
    {
    return (SPINN_MEM_UNAVAIL);
    }
  }

  // allocate memory for unit outputs
  if ((w_outputs[0] = ((activation_t *)
         spin1_malloc (wcfg.num_rows * sizeof (activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((w_outputs[1] = ((activation_t *)
         spin1_malloc (wcfg.num_rows * sizeof (activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for link deltas
  if ((w_link_deltas = ((long_delta_t * *)
         spin1_malloc (wcfg.num_rows * sizeof (long_delta_t *)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    if ((w_link_deltas[i] = ((long_delta_t *)
         spin1_malloc (wcfg.num_cols * sizeof (long_delta_t)))) == NULL
       )
    {
    return (SPINN_MEM_UNAVAIL);
    }
  }

  // allocate memory for errors
  if ((w_errors = ((error_t *)
         spin1_malloc (wcfg.num_rows * sizeof (error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((w_pkt_queue.queue = ((packet_t *)
         spin1_malloc (SPINN_WEIGHT_PQ_LEN * sizeof (packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  //TODO: the following memory allocation is to be used to store
  // the history of this set of values. When training
  // continuous networks, this history always needs to be saved.
  // For non-continuous networks, they only need to be stored if the
  // backpropTicks field of the network is greater than one. This
  // information needs to come in the wcfg structure.

  // allocate memory in SDRAM for output history
  if ((w_output_history = ((activation_t *)
         sark_xalloc (sv->sdram_heap,
                      wcfg.num_rows * ncfg.global_max_ticks * sizeof (activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// initialise variables
// ------------------------------------------------------------------------
void var_init (uint init_weights, uint reset_examples)
{
  // initialise weights from SDRAM if requested
  if (init_weights)
  {
    //NOTE: could use DMA
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      spin1_memcpy (w_weights[i],
                     &wt[i * wcfg.num_cols],
                     wcfg.num_cols * sizeof (weight_t)
                   );
    }
  }

#ifdef DEBUG_WEIGHTS
  for (uint r = 0; r < wcfg.num_rows; r++)
  {
    for (uint c =0; c < wcfg.num_cols; c++)
    {
      io_printf (IO_BUF, "w[%u][%u]: %k\n", r, c, w_weights[r][c]);
    }
  }
#endif

  // reset example index if requested
  //TODO: alternative algorithms for choosing example order!
  if (reset_examples)
  {
    example_inx = 0;
  }

  // initialise example counter
  example_cnt = 0;

  // initialise epoch
  epoch = 0;

  // initialise event id, number of events and event index
  evt        = 0;
  num_events = ex[example_inx].num_events;
  event_idx  = ex[example_inx].ev_idx;

  // initialise phase
  phase = SPINN_FORWARD;

  // initialise tick
  tick = SPINN_W_INIT_TICK;

  // initialise sync flags
  sync_rdy = FALSE;
  epoch_rdy = FALSE;
  net_stop_rdy = FALSE;
  net_stop = 0;

  // initialise unit outputs, link deltas, weight changes
  // error dot products and output history for tick 0
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    w_outputs[0][i] = wcfg.initOutput;

    for (uint j = 0; j < wcfg.num_cols; j++)
    {
      w_link_deltas[i][j] = 0;
      w_wchanges[i][j] = 0;
    }

    w_errors[i] = 0;
    w_output_history[i] = 0;
  }

  // initialise delta scaling factor
  // s15.16
  w_delta_dt = (1 << SPINN_FPREAL_SHIFT) / ncfg.ticks_per_int;

  // initialise pointers to received unit outputs
  wf_procs = 0;
  wf_comms = 1;

  // initialise thread semaphores
  wf_thrds_pend = SPINN_WF_THRDS;
  wb_thrds_pend = SPINN_WB_THRDS; // no link delta sum until last BP tick

  // initialise processing thread flag
  wb_active = FALSE;

  // initialise arrival scoreboards
  wf_arrived = 0;
  wb_arrived = 0;

  // initialise packet queue
  w_pkt_queue.head = 0;
  w_pkt_queue.tail = 0;

  // set weight update function
  wb_update_func = w_update_procs[xcfg.update_function];

  // initialise packet keys
  //NOTE: colour is initialised to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY(SPINN_FORWARD)
      | SPINN_BLOCK_KEY(wcfg.col_blk);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY(SPINN_BACKPROP)
      | SPINN_BLOCK_KEY(wcfg.row_blk);
  ldsaKey = rt[LDS] | SPINN_LDSA_KEY | SPINN_PHASE_KEY(SPINN_BACKPROP);

#ifdef DEBUG
  // ------------------------------------------------------------------------
  // DEBUG variables
  // ------------------------------------------------------------------------
  pkt_sent = 0;  // total packets sent
  sent_fwd = 0;  // packets sent in FORWARD phase
  sent_bkp = 0;  // packets sent in BACKPROP phase
  pkt_recv = 0;  // total packets received
  recv_fwd = 0;  // packets received in FORWARD phase
  recv_bkp = 0;  // packets received in BACKPROP phase
  pkt_fwbk = 0;  // unused packets received in FORWARD phase
  pkt_bwbk = 0;  // unused packets received in BACKPROP phase
  spk_recv = 0;  // sync packets received
  stp_sent = 0;  // stop packets sent
  stp_recv = 0;  // stop packets received
  stn_recv = 0;  // network_stop packets received
  lda_sent = 0;  // partial link_delta packets sent
  ldr_recv = 0;  // link_delta packets received
  wrng_fph = 0;  // FORWARD packets received in wrong phase
  wrng_bph = 0;  // BACKPROP received in wrong phase
  wght_ups = 0;  // number of weight updates done
  wrng_pth = 0;  // unexpected processing thread
  wrng_cth = 0;  // unexpected comms thread
  wrng_sth = 0;  // unexpected stop thread
  tot_tick = 0;  // total number of ticks executed
  // ------------------------------------------------------------------------
#endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// load stage configuration from SDRAM
// ------------------------------------------------------------------------
void stage_init (void)
{
  // clear output from previous stage
  sark_io_buf_reset();

  // initialise stage configuration from SDRAM
  spin1_memcpy (&xcfg, xadr, sizeof (stage_conf_t));

#ifdef DEBUG
  io_printf (IO_BUF, "stage %u configured\n", xcfg.stage_id);
  if (xcfg.training)
  {
    io_printf (IO_BUF, "train (updates:%u)\n", xcfg.num_epochs);
  }
  else
  {
    io_printf (IO_BUF, "test (examples:%u)\n", xcfg.num_examples);
  }
#endif

  // initialise variables for this stage (do NOT initialise weights)
  var_init (FALSE, xcfg.reset);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stage start callback: get stage started
// ------------------------------------------------------------------------
void stage_start (void)
{
#ifdef DEBUG
  // start log
  io_printf (IO_BUF, "----------------\n");
  io_printf (IO_BUF, "starting stage %u\n", xcfg.stage_id);
#endif

  // trigger computation, when execution starts
  spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// check exit code and print details of the state
// ------------------------------------------------------------------------
void stage_done (uint ec, uint key)
{
#if !defined(DEBUG) && !defined(DEBUG_EXIT)
  //NOTE: parameter 'key' is used only in DEBUG reporting
  (void) key;
#endif

  // pause timer and setup next stage,
  simulation_handle_pause_resume (stage_init);

#if defined(DEBUG) || defined(DEBUG_EXIT)
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
      io_printf (IO_BUF, "k:0x%0x\n", key);
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_TIMEOUT_EXIT:
      io_printf (IO_BUF, "timeout (h:%u e:%u p:%u t:%u) - abort!\n",
                 epoch, example_cnt, phase, tick
                );
      io_printf (IO_BUF, "(fp:%u  fc:%u)\n", wf_procs, wf_comms);
      io_printf (IO_BUF, "(fptd:%u bptd:%u)\n", wf_thrds_pend, wb_thrds_pend);
      io_printf (IO_BUF, "(fa:%u/%u ba:%u/%u)\n",
                 wf_arrived, wcfg.num_rows, wb_arrived, wcfg.num_cols
                );
      io_printf (IO_BUF, "stage aborted\n");
      break;
  }
#endif

#ifdef DEBUG
  // report diagnostics
  io_printf (IO_BUF, "total ticks:%d\n", tot_tick);
  io_printf (IO_BUF, "total recv:%d\n", pkt_recv);
  io_printf (IO_BUF, "total sent:%d\n", pkt_sent);
  io_printf (IO_BUF, "recv: fwd:%d bkp:%d\n", recv_fwd, recv_bkp);
  io_printf (IO_BUF, "sent: fwd:%d bkp:%d\n", sent_fwd, sent_bkp);
  io_printf (IO_BUF, "unused recv: fwd:%d bkp:%d\n", pkt_fwbk, pkt_bwbk);
  io_printf (IO_BUF, "ldsa sent:%d\n", lda_sent);
  io_printf (IO_BUF, "ldsr recv:%d\n", ldr_recv);
  io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
  io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
  io_printf (IO_BUF, "sync recv:%d\n", spk_recv);
  if (wrng_fph) io_printf (IO_BUF, "fwd wrong phase:%d\n", wrng_fph);
  if (wrng_bph) io_printf (IO_BUF, "bkp wrong phase:%d\n", wrng_bph);
  if (wrng_pth) io_printf (IO_BUF, "wrong pth:%d\n", wrng_pth);
  if (wrng_cth) io_printf (IO_BUF, "wrong cth:%d\n", wrng_cth);
  if (wrng_sth) io_printf (IO_BUF, "wrong sth:%d\n", wrng_sth);
  io_printf (IO_BUF, "------\n");
  io_printf (IO_BUF, "weight updates:%d\n", wght_ups);
#endif

#ifdef DEBUG
  // close log,
  io_printf (IO_BUF, "stopping stage %u\n", xcfg.stage_id);
  io_printf (IO_BUF, "----------------\n");
#endif

  // and let host know that we're done
  if (ec == SPINN_NO_ERROR) {
    simulation_ready_to_read ();
  } else {
    rt_error (RTE_SWERR);
  }
}
// ------------------------------------------------------------------------
