// SpiNNaker API
#include "spin1_api.h"

// front-end-common
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"
#include "init_t.h"
#include "comms_t.h"


// ------------------------------------------------------------------------
// threshold core initialisation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// load configurations from SDRAM
// ------------------------------------------------------------------------
uint cfg_init (void)
{
  io_printf (IO_BUF, "threshold\n");

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
  spin1_memcpy (&tcfg, dt, sizeof (t_conf_t));

  // inputs
  if (tcfg.input_grp)
  {
    it = (activation_t *) data_specification_get_region
      (INPUTS, data);
  }

  // targets
  if (tcfg.output_grp)
  {
    tt = (activation_t *) data_specification_get_region
      (TARGETS, data);
  }

  // example set
  es = (mlp_set_t *) data_specification_get_region
      (EXAMPLE_SET, data);

#ifdef DEBUG_CFG5
  io_printf (IO_BUF, "ne: %u\n", es->num_examples);
  io_printf (IO_BUF, "mt: %f\n", es->max_time);
  io_printf (IO_BUF, "nt: %f\n", es->min_time);
  io_printf (IO_BUF, "gt: %f\n", es->grace_time);
  io_printf (IO_BUF, "NaN: 0x%08x%\n", SPINN_FP_NaN);
#endif

  // examples
  ex = (mlp_example_t *) data_specification_get_region
      (EXAMPLES, data);

#ifdef DEBUG_CFG5
  for (uint i = 0; i < es->num_examples; i++)
  {
    io_printf (IO_BUF, "nx[%u]: %u\n", i, ex[i].num);
    io_printf (IO_BUF, "nv[%u]: %u\n", i, ex[i].num_events);
    io_printf (IO_BUF, "vi[%u]: %u\n", i, ex[i].ev_idx);
    io_printf (IO_BUF, "xf[%u]: %f\n", i, ex[i].freq);
  }
#endif

  // events
  ev = (mlp_event_t *) data_specification_get_region
      (EVENTS, data);

#ifdef DEBUG_CFG5
  uint evi = 0;
  for (uint i = 0; i < es->num_examples; i++)
  {
    for (uint j = 0; j < ex[i].num_events; j++)
    {
      io_printf (IO_BUF, "mt[%u][%u]: %f\n", i, j, ev[evi].max_time);
      io_printf (IO_BUF, "nt[%u][%u]: %f\n", i, j, ev[evi].min_time);
      io_printf (IO_BUF, "gt[%u][%u]: %f\n", i, j, ev[evi].grace_time);
      io_printf (IO_BUF, "ii[%u][%u]: %u\n", i, j, ev[evi].it_idx);
      evi++;
    }
  }
#endif

  // routing keys
  rt = (uint *) data_specification_get_region
      (ROUTING, data);

  // initialise stage configuration from SDRAM
  xadr = data_specification_get_region (STAGE, data);
  spin1_memcpy (&xcfg, xadr, sizeof (stage_conf_t));
  io_printf (IO_BUF, "stage %u configured\n", xcfg.stage_id);
  if (xcfg.training)
  {
    io_printf (IO_BUF, "train ", xcfg.num_examples);
  }
  else
  {
    io_printf (IO_BUF, "test ");
  }
  io_printf (IO_BUF, "for examples: %u\n", xcfg.num_examples);

#ifdef DEBUG_CFG
  io_printf (IO_BUF, "og: %d\n", tcfg.output_grp);
  io_printf (IO_BUF, "ig: %d\n", tcfg.input_grp);
  io_printf (IO_BUF, "nu: %d\n", tcfg.num_units);
  io_printf (IO_BUF, "fs: %d\n", tcfg.fwd_sync_expected);
  io_printf (IO_BUF, "bs: %d\n", tcfg.bkp_sync_expected);
  io_printf (IO_BUF, "wo: %d\n", tcfg.write_out);
  io_printf (IO_BUF, "wb: %d\n", tcfg.write_blk);
  io_printf (IO_BUF, "ie: %d\n", tcfg.out_integr_en);
  io_printf (IO_BUF, "dt: %f\n", tcfg.out_integr_dt);
  io_printf (IO_BUF, "np: %d\n", tcfg.num_out_procs);
  io_printf (IO_BUF, "p0: %d\n", tcfg.procs_list[0]);
  io_printf (IO_BUF, "p1: %d\n", tcfg.procs_list[1]);
  io_printf (IO_BUF, "p2: %d\n", tcfg.procs_list[2]);
  io_printf (IO_BUF, "p3: %d\n", tcfg.procs_list[3]);
  io_printf (IO_BUF, "p4: %d\n", tcfg.procs_list[4]);
  io_printf (IO_BUF, "wc: %f\n", tcfg.weak_clamp_strength);
  io_printf (IO_BUF, "io: %f\n", SPINN_LCONV_TO_PRINT(
        tcfg.initOutput, SPINN_ACTIV_SHIFT));
  io_printf (IO_BUF, "gc: %k\n", tcfg.group_criterion);
  io_printf (IO_BUF, "cf: %d\n", tcfg.criterion_function);
  io_printf (IO_BUF, "fg: %d\n", tcfg.is_first_output_group);
  io_printf (IO_BUF, "lg: %d\n", tcfg.is_last_output_group);
  io_printf (IO_BUF, "ef: %d\n", tcfg.error_function);
  io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
  io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
  io_printf (IO_BUF, "sk: 0x%08x\n", rt[STP]);
#endif

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory in DTCM and SDRAM
// ------------------------------------------------------------------------
uint mem_init (void)
{
  // allocate memory for nets -- stored in FORWARD phase for use in BACKPROP
  if ((t_nets = ((net_t *)
         spin1_malloc (tcfg.num_units * sizeof (net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for outputs
  if ((t_outputs = ((activation_t *)
         spin1_malloc (tcfg.num_units * sizeof (activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for output derivatives (equal to error derivative)
  if ((t_output_deriv = ((long_deriv_t *)
         spin1_malloc (tcfg.num_units * sizeof (long_deriv_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for deltas
  if ((t_deltas = ((delta_t *)
         spin1_malloc (tcfg.num_units * sizeof (delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((t_errors[0] = ((error_t *)
         spin1_malloc (tcfg.num_units * sizeof (error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((t_errors[1] = ((error_t *)
         spin1_malloc (tcfg.num_units * sizeof (error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for net packet queue
  // TODO: use correct length!
  if ((t_net_pkt_q.queue = ((packet_t *)
         spin1_malloc (SPINN_THLD_PQ_LEN * sizeof (packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for forward keys (one per partition)
  if ((t_fwdKey = ((uint *)
         spin1_malloc (tcfg.partitions * sizeof (uint)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for OUTPUT functions
  for (uint i = 0; i < tcfg.num_out_procs; i++)
  {
    if (t_init_out_procs[tcfg.procs_list[i]] != NULL)
    {
      // call the appropriate routine for pipeline initialisation
      uint exit_code  = t_init_out_procs[tcfg.procs_list[i]]();
      if (exit_code != SPINN_NO_ERROR)
        return (exit_code);
    }
  }

  // TODO: the following memory allocations are to be used to store
  // the histories of these sets of values. When training
  // continuous networks, these histories always need to be saved.
  // For non-continuous networks, they only need to be stored if the
  // backpropTicks field of the network is greater than one. This
  // information needs to come in the tcfg structure.

  // allocate memory in SDRAM for target history
  if ((t_target_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks
           * sizeof (activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for output derivative history
  if ((t_output_deriv_history = ((long_deriv_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks
           * sizeof (long_deriv_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for net history
  if ((t_net_history = ((net_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (net_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for output history
  if ((t_output_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks
           * sizeof (activation_t),
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
// initialise unit outputs and OUTPUT INTEGRATOR state
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//NOTE: There is a conflict in the initialisation routine between
// versions 2.63 and 2.64 of LENS. This function LENS version 2.63.
// Added comments indicate changes to apply LENS version 2.64
// ------------------------------------------------------------------------
void t_init_outputs (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "t_init_outputs\n");
#endif

  // initialise every unit output and send for processing
  for (uint i = 0; i < tcfg.num_units; i++)
  {
    // setup the initial output value.
    // Lens has two ways of initialise the output value,
    // as defined in Lens 2.63 and Lens 2.64,
    // and the two ways are not compatible

    // use initial values,
    // TODO: need to verify initInput with Lens
    // NOTE: The following code follows the output of Lens 2.63:
    // initialise the output value of the units

    t_outputs[i] = tcfg.initOutput;

    // if the OUTPUT INTEGRATOR is used
    // reset the array of the last values
    if (tcfg.out_integr_en) {
      t_last_integr_output[i] = tcfg.initOutput;

      t_last_integr_output_deriv[i] = 0;
    }

#ifdef DEBUG_CFG3
    io_printf (IO_BUF, "to[%u]: 0x%08x\n", i, t_outputs[i]);
#endif

    // and send unit output to weight cores
    while (!spin1_send_mc_packet ((t_fwdKey[i >> SPINN_BLOCK_SHIFT] | i),
                                   (uint) t_outputs[i],
                                   WITH_PAYLOAD
                                 )
          );

#ifdef DEBUG
    pkt_sent++;
    sent_fwd++;
#endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory for OUTPUT INTEGRATOR state
// ------------------------------------------------------------------------
uint init_out_integr ()
{
#ifdef TRACE_VRB
  io_printf (IO_BUF, "init_out_integr\n");
#endif

  // allocate memory for INTEGRATOR state
  //NOTE: these variables are initialised in function init_outputs ()
  if ((t_last_integr_output = ((activation_t *)
       spin1_malloc (tcfg.num_units * sizeof (activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((t_last_integr_output_deriv = ((long_deriv_t *)
       spin1_malloc (tcfg.num_units * sizeof (long_deriv_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((t_instant_outputs = ((activation_t *)
       spin1_malloc (tcfg.num_units * ncfg.global_max_ticks *
           sizeof (activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory for HARD CLAMP state
//NOTE: This function is currently a stub
// ------------------------------------------------------------------------
uint init_out_hard_clamp ()
{
#ifdef TRACE_VRB
  io_printf (IO_BUF, "init_out_hard_clamp\n");
#endif

/*
  if (xcfg.training)
  {
    // allocate memory for outputs
    if ((t_out_hard_clamp_data = ((short_activ_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (short_activ_t),
                       0, ALLOC_LOCK)
                       )) == NULL
       )
    {
      return (SPINN_MEM_UNAVAIL);
    }
  }

  io_printf (IO_BUF, "hc store addr %08x\n", (uint) t_out_hard_clamp_data);
*/

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory for WEAK CLAMP state
//NOTE: This function is currently a stub
// ------------------------------------------------------------------------
uint init_out_weak_clamp ()
{
#ifdef TRACE_VRB
  io_printf (IO_BUF, "init_out_weak_clamp\n");
#endif

/*
  if (xcfg.training)
  {
    // allocate memory for outputs
    if ((t_out_weak_clamp_data = ((short_activ_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (short_activ_t),
                       0, ALLOC_LOCK)
                       )) == NULL
       )
    {
      return (SPINN_MEM_UNAVAIL);
    }
  }
*/

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// initialise variables
// ------------------------------------------------------------------------
void var_init (void)
{
  // initialise epoch, example and event counters
  //TODO: alternative algorithms for choosing example order!
  epoch   = 0;
  example = 0;
  evt     = 0;

  // initialise phase
  phase = SPINN_FORWARD;

  // initialise number of events and event index
  num_events = ex[example].num_events;
  event_idx  = ex[example].ev_idx;

  // initialise example and event ticks
  tick = SPINN_T_INIT_TICK;
  ev_tick = SPINN_T_INIT_TICK;

  // initialise max and min ticks
  if (tcfg.is_last_output_group)
  {
    // get max number of ticks for first event
    if (ev[event_idx].max_time != SPINN_FP_NaN)
      max_ticks = (((ev[event_idx].max_time + SPINN_SMALL_VAL)
        * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;
    else
      max_ticks = (((es->max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;

    // get min number of ticks for first event
    if (ev[event_idx].min_time != SPINN_FP_NaN)
      min_ticks = (((ev[event_idx].min_time + SPINN_SMALL_VAL)
        * ncfg.ticks_per_int)
                    + (1 << (SPINN_FPREAL_SHIFT - 1)))
                    >> SPINN_FPREAL_SHIFT;
    else
      min_ticks = (((es->min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                    + (1 << (SPINN_FPREAL_SHIFT - 1)))
                    >> SPINN_FPREAL_SHIFT;
  }

  // initialise output derivatives, deltas and errors
  for (uint i = 0; i < tcfg.num_units; i++)
  {
    t_output_deriv[i] = 0;
    t_deltas[i] = 0;
    t_errors[0][i] = 0;
    t_errors[1][i] = 0;
  }

  // initialise pointers to received errors
  tb_procs = 0;
  tb_comms = 1;

  // initialise received net and error scoreboards
  tf_arrived = 0;
  tb_arrived = 0;

  // initialise synchronisation semaphores
  tf_thrds_pend = 1;
  tb_thrds_pend = 1;

  // initialise stop function and related flags
  if (tcfg.output_grp)
  {
    tf_stop_func = t_stop_procs[tcfg.criterion_function];
    tf_stop_crit = TRUE;
    tf_group_crit = TRUE;
    tf_event_crit = TRUE;
    tf_example_crit = TRUE;

    // variables for stop criterion computation
    t_max_output_unit = -1;
    t_max_target_unit = -1;
    t_max_output = SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT
               - SPINN_SHORT_ACTIV_SHIFT);
    t_max_target = SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT
               - SPINN_SHORT_ACTIV_SHIFT);

    // no need to wait for previous value if first in chain
    if (tcfg.is_first_output_group)
    {
      tf_initChain = 1;
      tf_chain_prev = TRUE;
    }
    else
    {
      tf_initChain = 0;
    }
    tf_chain_rdy = tf_initChain;

    if (tcfg.is_last_output_group)
    {
      // "broadcast" key
      tf_stop_key = rt[STP] | SPINN_STOP_KEY;

      // "stop final" key
      tf_stpn_key = rt[STP] | SPINN_STPN_KEY;
    }
    else
    {
      // "daisy chain" key
      tf_stop_key = rt[STP] | SPINN_STPC_KEY;
    }
  }

#ifdef DEBUG_VRB
  io_printf (IO_BUF, "tsk = 0x%08x\n", tf_stop_key);
#endif

  // initialise processing thread flag
  t_active = FALSE;

  // initialise received sync packets scoreboard
  t_sync_arrived = 0;

  // initialise sync packets flag
  t_sync_rdy = FALSE;

  // initialise net packet queue
  t_net_pkt_q.head = 0;
  t_net_pkt_q.tail = 0;

#ifdef DEBUG_VRB
  io_printf (IO_BUF, "wo:%d\n", tcfg.write_out);
#endif

  // check if writing outputs to host
  if (tcfg.write_out)
  {
    // initialise SDP message buffer,
    // Fill in SDP destination fields
    t_sdp_msg.tag = SPINN_SDP_IPTAG;      // IPTag
    t_sdp_msg.dest_port = PORT_ETH;       // Ethernet
    t_sdp_msg.dest_addr = sv->dbg_addr;   // Root chip

    // Fill in SDP source & flag fields,
    t_sdp_msg.flags = SPINN_SDP_FLAGS;
    t_sdp_msg.srce_port = coreID;
    t_sdp_msg.srce_addr = sv->p2p_addr;

    // compute total ticks in first example -- info to be sent to host,
    //TODO: cannot compute correctly -- variable if completion criteria used
    t_tot_ticks = 0;
    for (uint i = 0; i < num_events; i++)
    {
      // update number of ticks for new event
      if (ev[event_idx + i].max_time != SPINN_FP_NaN)
      {
        t_tot_ticks += (((ev[event_idx + i].max_time + SPINN_SMALL_VAL)
       * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
      }
      else
      {
        t_tot_ticks += (((es->max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
      }
    }

    // and limit to the global maximum if required
    if (t_tot_ticks > ncfg.global_max_ticks - 1)
    {
      t_tot_ticks = ncfg.global_max_ticks - 1;
    }
  }

  // initialise packet keys
  //NOTE: colour is initialised to 0
  for (uint p = 0; p < tcfg.partitions; p++) {
    t_fwdKey[p] = rt[FWDT + p] | SPINN_PHASE_KEY (SPINN_FORWARD);
  }
  bkpKey = rt[BKP] | SPINN_PHASE_KEY (SPINN_BACKPROP);

  // if input or output group initialise event input/target index
  if (tcfg.input_grp || tcfg.output_grp)
  {
    t_it_idx = ev[event_idx].it_idx * tcfg.num_units;
  }

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
spk_sent = 0;  // sync packets sent
spk_recv = 0;  // sync packets received
chn_sent = 0;  // chain packets sent
chn_recv = 0;  // chain packets received
stp_sent = 0;  // stop packets sent
stp_recv = 0;  // stop packets received
stn_sent = 0;  // network_stop packets sent
stn_recv = 0;  // network_stop packets received
wrng_phs = 0;  // packets received in wrong phase
wrng_tck = 0;  // FORWARD packets received in wrong tick
wrng_btk = 0;  // BACKPROP packets received in wrong tick
wght_ups = 0;  // number of weight updates done
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
  // clear output from earlier runs
  sark_io_buf_reset();

  // initialise stage configuration from SDRAM
  spin1_memcpy (&xcfg, xadr, sizeof (stage_conf_t));
  io_printf (IO_BUF, "stage %u configured\n", xcfg.stage_id);
  if (xcfg.training)
  {
    io_printf (IO_BUF, "train ", xcfg.num_examples);
  }
  else
  {
    io_printf (IO_BUF, "test ");
  }
  io_printf (IO_BUF, "for examples: %u\n", xcfg.num_examples);

  // re-initialise variables for this stage
  var_init ();
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stage start callback: get stage started
// ------------------------------------------------------------------------
void stage_start (void)
{
  // start log,
  io_printf (IO_BUF, "----------------\n");
  io_printf (IO_BUF, "starting stage %u\n", xcfg.stage_id);

  // send initial outputs to host -- if required,
  if (tcfg.write_out)
  {
    spin1_schedule_callback (send_info_to_host, 0, 0, SPINN_T_SEND_OUTS_P);
  }

  // and send initial outputs to w cores -- when simulation starts
  spin1_schedule_callback (t_init_outputs, 0, 0, SPINN_T_INIT_OUT_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// check exit code and print details of the state
// ------------------------------------------------------------------------
void stage_done (uint ec)
{
  // pause timer and setup next stage,
  simulation_handle_pause_resume (stage_init);

  // report problems -- if any
  switch (ec)
  {
    case SPINN_NO_ERROR:
      tick = 0;  // adjust tick for output reporting to host (below)
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
      io_printf (IO_BUF, "(tactive:%u ta:%u/%u tb:%u/%u)\n",
                  t_active, tf_arrived, tcfg.num_units,
                  tb_arrived, tcfg.num_units
                );
      io_printf (IO_BUF, "(tsr:%u tsa:%u/%u)\n",
                  t_sync_rdy, t_sync_arrived, tcfg.fwd_sync_expected
                );
      io_printf (IO_BUF, "(tcr:%u)\n",
                  tf_chain_rdy
                );
#endif
      break;
  }

#ifdef DEBUG
  // report diagnostics,
  io_printf (IO_BUF, "total ticks:%d\n", tot_tick);
  io_printf (IO_BUF, "total recv:%d\n", pkt_recv);
  io_printf (IO_BUF, "total sent:%d\n", pkt_sent);
  io_printf (IO_BUF, "recv: fwd:%d bkp:%d\n", recv_fwd, recv_bkp);
  io_printf (IO_BUF, "sent: fwd:%d bkp:%d\n", sent_fwd, sent_bkp);
  io_printf (IO_BUF, "sync recv:%d\n", spk_recv);
  if (tcfg.is_first_output_group)
  {
    io_printf (IO_BUF, "chain recv: first\n");
  }
  else
  {
  io_printf (IO_BUF, "chain recv:%d\n", chn_recv);
  }
  if (tcfg.is_last_output_group)
  {
    io_printf (IO_BUF, "stop sent:%d\n", stp_sent);
    io_printf (IO_BUF, "stpn sent:%d\n", stn_sent);
  }
  else
  {
    io_printf (IO_BUF, "chain sent:%d\n", chn_sent);
    io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
    io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
  }
  if (wrng_phs) io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
  if (wrng_tck) io_printf (IO_BUF, "wrong tick:%d\n", wrng_tck);
  if (wrng_btk) io_printf (IO_BUF, "wrong btick:%d\n", wrng_btk);
#endif

  // send final outputs to host,
  if (tcfg.write_out)
  {
    send_outputs_to_host (SPINN_HOST_FINAL, tick);
  }

  // close log,
  io_printf (IO_BUF, "stopping stage %u\n", xcfg.stage_id);
  io_printf (IO_BUF, "----------------\n");

  // and let host know that we're done
  if (ec == SPINN_NO_ERROR) {
    simulation_ready_to_read ();
  } else {
    rt_error (RTE_SWERR);
  }
}
// ------------------------------------------------------------------------
