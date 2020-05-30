// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_i.h"
#include "comms_i.h"


// ------------------------------------------------------------------------
// input core initialisation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// load configurations from SDRAM
// ------------------------------------------------------------------------
uint cfg_init (void)
{
  io_printf (IO_BUF, "input\n");

  // read the data specification header
  data_specification_metadata_t * data =
          data_specification_get_data_address();
  if (!data_specification_read_header (data))
  {
    return (SPINN_CFG_UNAVAIL);
  }

  // set up the simulation interface (system region)
  //NOTE: these variables are not used!
  uint32_t n_steps, run_forever, step;
  if (!simulation_steps_initialise(
      data_specification_get_region(SYSTEM, data),
      APPLICATION_NAME_HASH, &n_steps, &run_forever, &step, 0, 0))
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
  spin1_memcpy (&icfg, dt, sizeof (i_conf_t));

  // inputs if this core receives inputs from examples file
  if (icfg.input_grp)
  {
    it = (activation_t *) data_specification_get_region
      (INPUTS, data);
  }

  // examples
  ex = (mlp_example_t *) data_specification_get_region
      (EXAMPLES, data);

  // events
  ev = (mlp_event_t *) data_specification_get_region
      (EVENTS, data);

  // routing keys
  rt = (uint *) data_specification_get_region
      (ROUTING, data);

  // stage configuration address
  address_t xt = data_specification_get_region (STAGE, data);

  // initialise network configuration from SDRAM
  spin1_memcpy (&xcfg, xt, sizeof (stage_conf_t));

#ifdef DEBUG_CFG0
  io_printf (IO_BUF, "og: %d\n", icfg.output_grp);
  io_printf (IO_BUF, "ig: %d\n", icfg.input_grp);
  io_printf (IO_BUF, "nu: %d\n", icfg.num_units);
  io_printf (IO_BUF, "np: %d\n", icfg.num_in_procs);
  io_printf (IO_BUF, "p0: %d\n", icfg.procs_list[0]);
  io_printf (IO_BUF, "p1: %d\n", icfg.procs_list[1]);
  io_printf (IO_BUF, "ie: %d\n", icfg.in_integr_en);
  io_printf (IO_BUF, "dt: %f\n", icfg.in_integr_dt);
  io_printf (IO_BUF, "sc: %f\n", icfg.soft_clamp_strength);
  io_printf (IO_BUF, "in: %d\n", icfg.initNets);
  io_printf (IO_BUF, "io: %f\n", SPINN_LCONV_TO_PRINT(
      icfg.initOutput, SPINN_ACTIV_SHIFT));
  io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
  io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
#endif

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

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory and initialise variables
// ------------------------------------------------------------------------
uint var_init (void)
{
  uint i;

  // allocate memory for nets
  if ((i_nets = ((long_net_t *)
         spin1_malloc (icfg.num_units * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for deltas
  if ((i_deltas = ((long_delta_t *)
         spin1_malloc (icfg.num_units * sizeof(long_delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // TODO: probably this variable can be removed
  // allocate memory to store delta values during the first BACKPROPagation tick
  if ((i_init_delta = ((long_delta_t *)
         spin1_malloc (icfg.num_units * sizeof(long_delta_t)))) == NULL
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

  // initialise tick
  //NOTE: input cores do not have a tick 0
  tick = SPINN_I_INIT_TICK;

  // initialise scoreboards
  if_done = 0;
  ib_done = 0;

  // initialise synchronisation semaphores
  if_thrds_pend = 1;

  // initialise processing thread flag
  i_active = FALSE;

  // initialise packet queue
  i_pkt_queue.head = 0;
  i_pkt_queue.tail = 0;

  // initialise packet keys
  // allocate memory for backprop keys (one per partition)
  if ((i_bkpKey = ((uint *)
         spin1_malloc (icfg.partitions * sizeof(uint)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  //NOTE: colour is initialised to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY(SPINN_FORWARD);
  for (uint p = 0; p < icfg.partitions; p++) {
	  i_bkpKey[p] = rt[BKPI + p] | SPINN_PHASE_KEY (SPINN_BACKPROP);
  }

  // if input or output group initialise event input/target index
  if (icfg.input_grp || icfg.output_grp)
  {
    i_it_idx = ev[event_idx].it_idx * icfg.num_units;
  }

  // if the network requires training and elements of the pipeline require
  // initialisation, then follow the appropriate procedure
  // use the list of procedures in use from lens and call the appropriate
  // initialisation routine from the i_init_in_procs function pointer list

  for (i = 0; i < icfg.num_in_procs; i++)
    if (i_init_in_procs[icfg.procs_list[i]] != NULL)
    {
      int return_value;
      // call the appropriate routine for pipeline initialisation
      return_value = i_init_in_procs[icfg.procs_list[i]]();

      // if return value contains error, return it
      if (return_value != SPINN_NO_ERROR)
          return return_value;
    }

  // allocate memory in SDRAM for input history,
  // TODO: this needs a condition on the requirement to have input history
  // which needs to come as a configuration parameter
  if ((i_net_history = ((long_net_t *)
          sark_xalloc (sv->sdram_heap,
                       icfg.num_units * ncfg.global_max_ticks * sizeof(long_net_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // and initialise net history for tick 0.
  //TODO: understand why the values for tick 0 are used!
  for (uint i = 0; i < icfg.num_units; i++)
  {
    i_net_history[i] = 0;
  }

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// load stage configuration from SDRAM
// ------------------------------------------------------------------------
void stage_init (void)
{
  // read the data specification header
  data_specification_metadata_t * data =
          data_specification_get_data_address();
  if (!data_specification_read_header (data))
  {
    // report results and abort simulation
    stage_done (SPINN_CFG_UNAVAIL);
  }

  // stage configuration address
  address_t xadr = data_specification_get_region (STAGE, data);

  // initialise network configuration from SDRAM
  spin1_memcpy (&xcfg, xadr, sizeof (stage_conf_t));

  io_printf (IO_BUF, "stage configured\n");
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
  simulation_handle_pause_resume (stage_init);

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
      io_printf (IO_BUF, "(fd:%u bd:%u)\n", if_done, ib_done);
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
  io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
  io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
  if (wrng_phs) io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
  if (wrng_tck) io_printf (IO_BUF, "wrong tick:%d\n", wrng_tck);
  if (wrng_btk) io_printf (IO_BUF, "wrong btick:%d\n", wrng_btk);
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
