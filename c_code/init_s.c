/*
 * Copyright (c) 2015-2021 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// SpiNNaker API
#include "spin1_api.h"

// front-end-common
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"
#include "init_s.h"
#include "comms_s.h"


// ------------------------------------------------------------------------
// sum core initialisation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// load configurations from SDRAM
// ------------------------------------------------------------------------
uint cfg_init (void)
{
#ifdef DEBUG
  io_printf (IO_BUF, "sum\n");
#endif

#ifdef PROFILE
  // configure timer 2 for profiling
  tc[T2_CONTROL] = SPINN_PROFILER_CFG;
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
  spin1_memcpy (&scfg, dt, sizeof (s_conf_t));

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
  io_printf (IO_BUF, "nu: %d\n", scfg.num_units);
  io_printf (IO_BUF, "fe: %d\n", scfg.fwd_expected);
  io_printf (IO_BUF, "be: %d\n", scfg.bkp_expected);
  io_printf (IO_BUF, "le: %d\n", scfg.lds_expected);
  io_printf (IO_BUF, "uf: %d\n", xcfg.update_function);
  io_printf (IO_BUF, "fg: %d\n", scfg.is_first_group);
  io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
  io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
  io_printf (IO_BUF, "lk: 0x%08x\n", rt[LDS]);
#endif

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory in DTCM and SDRAM
// ------------------------------------------------------------------------
uint mem_init (void)
{
  // allocate memory for nets
  if ((s_nets[0] = ((long_net_t *)
         spin1_malloc (scfg.num_units * sizeof (long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_nets[1] = ((long_net_t *)
         spin1_malloc (scfg.num_units * sizeof (long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((s_errors[0] = ((long_error_t *)
         spin1_malloc (scfg.num_units * sizeof (long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_errors[1] = ((long_error_t *)
         spin1_malloc (scfg.num_units * sizeof (long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((s_pkt_queue.queue = ((packet_t *)
         spin1_malloc (SPINN_SUM_PQ_LEN * sizeof (packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received net b-d-ps scoreboards
  if ((sf_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof (scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sf_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof (scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received error b-d-ps scoreboards
  if ((sb_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof (scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sb_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof (scoreboard_t)))) == NULL
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
void var_init (uint reset_examples)
{
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
  //NOTE: SUM cores do not have a tick 0
  tick = SPINN_S_INIT_TICK;

  // initialise network stop flag
  net_stop_rdy = FALSE;
  net_stop = 0;

  // initialise nets, errors and scoreboards
  for (uint i = 0; i < scfg.num_units; i++)
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

  // initialise thread semaphores
  sf_thrds_pend = SPINN_SF_THRDS;
  sb_thrds_pend = SPINN_SB_THRDS;

  // initialise processing thread flag
  s_active = FALSE;

  // initialise lds
  s_lds_part = 0;
  s_lds_arrived = 0;

  // initialise packet queue
  s_pkt_queue.head = 0;
  s_pkt_queue.tail = 0;

  // initialise packet keys
  //NOTE: colour is implicitly initialised to 0
  fwdKey = rt[FWD] | SPINN_PHASE_KEY (SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY (SPINN_BACKPROP);
  ldsKey = rt[LDS] | SPINN_LDSA_KEY | SPINN_PHASE_KEY (SPINN_BACKPROP);
  fdsKey = rt[FDS] | SPINN_SYNC_KEY | SPINN_PHASE_KEY (SPINN_FORWARD);

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
  stp_sent = 0;  // stop packets sent
  stp_recv = 0;  // stop packets received
  stn_recv = 0;  // network_stop packets received
  lds_recv = 0;  // link_delta packets received
  lds_sent = 0;  // link_delta packets sent
  wrng_phs = 0;  // packets received in wrong phase
  wrng_pth = 0;  // unexpected processing thread
  wrng_cth = 0;  // unexpected comms thread
  wrng_sth = 0;  // unexpected stop thread
  tot_tick = 0;  // total number of ticks executed
  // ------------------------------------------------------------------------
#endif

#ifdef PROFILE
// ------------------------------------------------------------------------
// PROFILER variables
// ------------------------------------------------------------------------
prf_fwd_min = SPINN_PROFILER_START;  // minimum FORWARD processing time
prf_fwd_max = 0;                     // maximum FORWARD processing time
prf_bkp_min = SPINN_PROFILER_START;  // minimum BACKPROP processing time
prf_bkp_max = 0;                     // maximum BACKPROP processing time
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

  // initialise variables for this stage
  var_init (xcfg.reset);
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
      io_printf (IO_BUF, "(fd:%u bd:%u)\n", sf_done, sb_done);
      for (uint i = 0; i < scfg.num_units; i++)
      {
        io_printf (IO_BUF, "%2d: (fa[0]:%u ba[0]:%u fa[1]:%u ba[1]:%u)\n", i,
                    sf_arrived[0][i], sb_arrived[0][i],
                    sf_arrived[1][i], sb_arrived[1][i]
                  );
      }
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
  io_printf (IO_BUF, "lds sent:%d\n", lds_sent);
  io_printf (IO_BUF, "lds recv:%d\n", lds_recv);
  io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
  io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
  io_printf (IO_BUF, "sync sent:%d\n", spk_sent);
  if (wrng_phs) io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
  if (wrng_pth) io_printf (IO_BUF, "wrong pth:%d\n", wrng_pth);
  if (wrng_cth) io_printf (IO_BUF, "wrong cth:%d\n", wrng_cth);
  if (wrng_sth) io_printf (IO_BUF, "wrong sth:%d\n", wrng_sth);
#endif

#ifdef PROFILE
  // report PROFILER values
  io_printf (IO_BUF, "min fwd proc:%u\n", prf_fwd_min);
  io_printf (IO_BUF, "max fwd proc:%u\n", prf_fwd_max);
  if (xcfg.training)
  {
    io_printf (IO_BUF, "min bkp proc:%u\n", prf_bkp_min);
    io_printf (IO_BUF, "max bkp proc:%u\n", prf_bkp_max);
  }
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
