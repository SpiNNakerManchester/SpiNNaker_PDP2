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
#include <recording.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"


// ------------------------------------------------------------------------
// threshold core communications routines
// includes functions to transfer data between DTCM and SDRAM
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// initial handling of received packets
// (FORWARD, BACKPROP, criterion, stop and net_stop types)
// ------------------------------------------------------------------------
void t_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

  // check packet phase,
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // BACKPROP-phase packets are handled immediately
  if (ph == SPINN_BACKPROP)
  {
    w_handleBKPPacket (key, payload);
    return;
  }

  // FORWARD-phase packets are queued for background processing
  uint new_tail = (t_pkt_queue.tail + 1) % SPINN_THLD_PQ_LEN;

  // check if space in packet queue,
  if (new_tail == t_pkt_queue.head)
  {
    // report queue full error
    stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full enqueue packet,
    t_pkt_queue.queue[t_pkt_queue.tail].key = key;
    t_pkt_queue.queue[t_pkt_queue.tail].payload = payload;
    t_pkt_queue.tail = new_tail;

    // and schedule FORWARD processing thread -- if not active already
    //TODO: do we need to check phase?
    if (!tf_active)
    {
      tf_active = TRUE;
      spin1_schedule_callback (t_processFWDQueue, 0, 0, SPINN_TF_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// handle BACKPROP-phase packets
// (BACKPROP type)
// ------------------------------------------------------------------------
void w_handleBKPPacket (uint key, uint payload)
{
  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process BACKPROP data packet,
  if (pkt_type == SPINN_DATA_KEY)
  {
    t_backprop_packet (key, payload);
    return;
  }

  // or process synchronisation packet,
  if (pkt_type == SPINN_BSGN_KEY)
  {
    t_bsgn_packet ();
    return;
  }

  // or process synchronisation packet,
  if (pkt_type == SPINN_SYNC_KEY)
  {
    t_sync_packet ();
    return;
  }

#ifdef DEBUG
  // or report unexpected packet type
  stage_done (SPINN_UNXPD_PKT, key);
#endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process FORWARD-phase packet queue until empty
// ------------------------------------------------------------------------
void t_processFWDQueue (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

  // access queue with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // process until queue empty
  while (t_pkt_queue.head != t_pkt_queue.tail)
  {
    // dequeue packet,
    uint key = t_pkt_queue.queue[t_pkt_queue.head].key;
    uint payload = (net_t) t_pkt_queue.queue[t_pkt_queue.head].payload;
    t_pkt_queue.head = (t_pkt_queue.head + 1) % SPINN_THLD_PQ_LEN;

    // restore interrupts after queue access,
    spin1_mode_restore (cpsr);

    // check packet type,
    uint pkt_type = key & SPINN_TYPE_MASK;

    // process FORWARD data packet,
    if (pkt_type == SPINN_DATA_KEY)
    {
      tf_process (key, payload);
    }

    // or process criterion packet,
    else if (pkt_type == SPINN_CRIT_KEY)
    {
      t_criterion_packet (key);
    }

    // or process forward sync gen packet,
    else if (pkt_type == SPINN_FSGN_KEY)
    {
      t_fsgn_packet ();
    }

    // or process tick stop packet,
    else if (pkt_type == SPINN_STOP_KEY)
    {
      t_stop_packet (key);
    }

    // or process network stop packet,
    else if (pkt_type == SPINN_STPN_KEY)
    {
      t_net_stop_packet (key);
    }

    // or process deadlock recovery packet,
    else if (pkt_type == SPINN_DLRV_KEY)
    {
#ifdef DEBUG
      dlr_recv++;
#endif

      if ((key & SPINN_DLRV_MASK) == SPINN_DLRV_ABT)
      {
        // report timeout error
        stage_done (SPINN_TIMEOUT_EXIT, 0);
      }
      else
      {
        t_dlrv_packet ();
      }
    }

#ifdef DEBUG
    // or report unexpected packet type,
    else
    {
      stage_done (SPINN_UNXPD_PKT, key);
    }
#endif

    // and access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // flag going to sleep,
  tf_active = FALSE;

  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a criterion packet
// ------------------------------------------------------------------------
void t_criterion_packet (uint key)
{
#ifdef DEBUG
  crt_recv++;
#endif

  // partial criterion value arrived,
  tf_crit_prev = tf_crit_prev && (key & SPINN_STPD_MASK);

  // update scoreboard,
  tf_crit_arrived++;

  // and check if all criterion packets arrived
  if (tf_crit_arrived == tcfg.crit_expected)
  {
    // initialise scoreboard for next tick,
    tf_crit_arrived = 0;

    // access thread semaphore with interrupts disabled,
    uint cpsr = spin1_int_disable ();

    // and check if all other threads are done
    if (tf_thrds_pend == SPINN_THRD_CRIT)
    {
      // if done initialise thread semaphore,
      tf_thrds_pend = tf_thrds_init;

      // restore interrupts after flag access,
      spin1_mode_restore (cpsr);

      // send stop packet,
      tf_send_stop ();

      // and advance tick if last_output_group
      //NOTE: last output group does not get a tick stop packet
      // so it's ready to advance tick
      if (tcfg.is_last_output)
      {
        tf_advance_tick ();
      }
    }
    else
    {
      // if not done report thread done,
      tf_thrds_pend &= ~SPINN_THRD_CRIT;

      // and restore interrupts after semaphore access
      spin1_mode_restore (cpsr);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a forward sync generation packet
// ------------------------------------------------------------------------
void t_fsgn_packet (void)
{
#ifdef DEBUG
  fsg_recv++;
  if (phase == SPINN_BACKPROP)
    wrng_fph++;
#endif

  // access thread semaphore with interrupts disabled,
  uint cpsr = spin1_int_disable();

  // and check if all other threads are done
  if (tf_thrds_pend == SPINN_THRD_FSGN)
  {
    // if done initialise thread semaphore,
    tf_thrds_pend = tf_thrds_init;

    // restore interrupts after flag access,
    spin1_mode_restore(cpsr);

    // send stop packet,
    tf_send_stop();

    // and advance tick if last_output_group
    //NOTE: last output group does not get a tick stop packet
    // so it's ready to advance tick
    if (tcfg.is_last_output)
    {
      tf_advance_tick();
    }
  }
  else
  {
    // if not done report thread done,
    tf_thrds_pend &= ~SPINN_THRD_FSGN;

    // and restore interrupts after semaphore access
    spin1_mode_restore(cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a tick stop packet
// ------------------------------------------------------------------------
void t_stop_packet (uint key)
{
#ifdef DEBUG
  stp_recv++;
  if (phase == SPINN_BACKPROP)
    wrng_fph++;
#endif

  // get tick stop decision,
  tick_stop = key & SPINN_STPD_MASK;

  // and advance tick
  tf_advance_tick ();
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop packet
// ------------------------------------------------------------------------
void t_net_stop_packet (uint key)
{
#ifdef DEBUG
  stn_recv++;
#endif

  // network stop decision arrived,
  net_stop = key & SPINN_STPD_MASK;

  // access flag with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // and check if ready for network stop decision
  if (net_stop_rdy)
  {
    // clear flag,
    net_stop_rdy = FALSE;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // and decide what to do
    if (net_stop)
    {
      // finish stage and report no error
      spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
    }
  }
  else
  {
    // flag ready for net_stop decision,
    net_stop_rdy = TRUE;

    // and restore interrupts after flag access
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a deadlock recovery packet
// ------------------------------------------------------------------------
void t_dlrv_packet (void)
{
  // restart tick
  if (phase == SPINN_FORWARD)
  {
    // initialise thread semaphore,
    tf_thrds_pend = tf_thrds_init;

    // initialise scoreboards,
    tf_arrived = 0;
    tf_crit_arrived = 0;

    // initialise previous value,
    tf_crit_prev = TRUE;

    // and initialise processing thread flag
    tf_active = FALSE;
  }
  else
  {
    // initialise thread semaphore,
    tb_thrds_pend = tb_thrds_init;

    // initialise scoreboard,
    tb_arrived = 0;

    // and trigger computation
    spin1_schedule_callback (tb_process, 0, 0, SPINN_TB_PROCESS_P);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a BACKPROP data packet
// ------------------------------------------------------------------------
void t_backprop_packet (uint key, uint payload)
{
#ifdef DEBUG
  recv_bkp++;
  if (phase == SPINN_FORWARD)
    wrng_bph++;
#endif

  // get error index: mask out phase, core and block data,
  uint inx = key & SPINN_ERROR_MASK;

  // store received error,
  t_errors[tb_comms][inx] = (error_t) payload;

  // update scoreboard,
  tb_arrived++;

  // if all expected errors have arrived may move to next tick
  if (tb_arrived == tcfg.num_units)
  {
    // initialise arrival scoreboard for next tick,
    tb_arrived = 0;

    // update pointer to received errors,
    tb_comms = 1 - tb_comms;

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(tb_thrds_pend & SPINN_THRD_COMS))
      wrng_cth++;
#endif

    // and check if all other threads are done,
    if (tb_thrds_pend == SPINN_THRD_COMS)
    {
      // if done initialise thread semaphore,
      tb_thrds_pend = tb_thrds_init;

      // advance tick,
      //NOTE: last t core does *not* get a sync packet
      if (tcfg.is_last_output)
      {
        spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_TB_TICK_P);
      }

      // and backprop sync packet to allow next tick to start
      while (!spin1_send_mc_packet (bpsKey, 0, NO_PAYLOAD));

#ifdef DEBUG
      pkt_sent++;
      if (tcfg.is_last_output) spk_sent++;
      else bsg_sent++;
#endif
    }
    else
    {
      // if not done report comms thread done
      tb_thrds_pend &= ~SPINN_THRD_COMS;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a bakprop sync generation packet
// ------------------------------------------------------------------------
void t_bsgn_packet (void)
{
#ifdef DEBUG
  bsg_recv++;
  if (phase == SPINN_FORWARD)
    wrng_bph++;
#endif

  // update scoreboard,
  tb_bsgn_arrived++;

  // and check if all backprop sync gen packets arrived
  if (tb_bsgn_arrived == tb_bsgn_expected)
  {
    // initialise scoreboard for next tick,
    tb_bsgn_arrived = 0;

    // check if all other threads done
    if (tb_thrds_pend == SPINN_THRD_BSGN)
    {
      // initialise semaphore,
      tb_thrds_pend = tb_thrds_init;

      // advance tick,
      //NOTE: last t core does *not* get a sync packet
      if (tcfg.is_last_output)
      {
        spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_TB_TICK_P);
      }

      // and send sync packet to allow next tick to start
      while (!spin1_send_mc_packet(bpsKey, 0, NO_PAYLOAD));

#ifdef DEBUG
      pkt_sent++;
      if (tcfg.is_last_output) spk_sent++;
      else bsg_sent++;
#endif
    }
    else
    {
      // report sync thread done
      tb_thrds_pend &= ~SPINN_THRD_BSGN;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a sync packet
// ------------------------------------------------------------------------
void t_sync_packet (void)
{
#ifdef DEBUG
  spk_recv++;
  if (phase == SPINN_FORWARD)
    wrng_bph++;
#endif

  // advance tick
  spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_TB_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// in the FORWARD phase the convergence criterion may require the simulation to
// stop before the maximum time is reached. This routine sends a broadcast
// message to communicate the final decision if the criterion has been reached
// across all the output groups to all the cores in the simulation
// ------------------------------------------------------------------------
void tf_send_stop (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "tf_send_stop\n");
#endif

  // "aggregate" criteria,
  tf_stop_crit = tf_stop_crit && tf_crit_prev;

  // initialise previous value,
  //TODO: should this be done in critical section?
  tf_crit_prev = TRUE;

  // make stop decision,
  if (tcfg.is_last_output)
  {
    tf_group_crit = tf_stop_crit;

    if (!xcfg.training)
    {
      t_test_results.examples_correct += tf_stop_crit && (ev_tick >=min_ticks);
    }

    tf_stop_crit = (ev_tick >= max_ticks)
                     || (tick == ncfg.global_max_ticks - 1)
                     || (tf_stop_crit && (ev_tick >= min_ticks));
    tick_stop = tf_stop_crit;
  }

  // FORWARD aggregated criterion,
  while (!spin1_send_mc_packet ((tf_stop_key | tf_stop_crit),
                                 0,
                                 NO_PAYLOAD
                               )
        );

#ifdef DEBUG
  pkt_sent++;
  if (tcfg.is_last_output) stp_sent++;
  else crt_sent++;
#endif

  // and initialise criterion for next tick
  tf_stop_crit = TRUE;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores the net of the specified unit for the current tick
// ------------------------------------------------------------------------
void store_net (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "store_net\n");
#endif

  t_net_history[(tick * tcfg.num_units) + inx] = t_nets[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores the net of the specified unit for the requested tick
// ------------------------------------------------------------------------
void restore_net (uint inx, uint tick)
{
#ifdef TRACE
    io_printf (IO_BUF, "restore_net\n");
#endif

  t_nets[inx] = t_net_history[((tick * tcfg.num_units) + inx)];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores the output of the specified unit for the current tick
// ------------------------------------------------------------------------
void store_output (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "store_output\n");
#endif

  t_output_history[(tick * tcfg.num_units) + inx] = t_outputs[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores the output of the specified unit for the requested tick
// ------------------------------------------------------------------------
void restore_output (uint inx, uint tick)
{
#ifdef TRACE
  io_printf (IO_BUF, "restore_output\n");
#endif

  t_outputs[inx] = t_output_history[((tick * tcfg.num_units) + inx)];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores the output derivative of the specified unit for the current tick
// ------------------------------------------------------------------------
void store_output_deriv (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "store_output_deriv\n");
#endif

  t_output_deriv_history[(tick * tcfg.num_units) + inx] = t_output_deriv[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores the output derivative of the specified unit for the requested tick
// ------------------------------------------------------------------------
void restore_output_deriv (uint inx, uint tick)
{
#ifdef TRACE
  io_printf (IO_BUF, "restore_output_deriv\n");
#endif

  t_output_deriv[inx] =
    t_output_deriv_history[(tick * tcfg.num_units) + inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// record outputs - to be picked up by the host
// ------------------------------------------------------------------------
void record_outputs (void)
{
  // cast outputs to the right size,
  short_activ_t outputs[tcfg.num_units];

  for (uint i = 0; i < tcfg.num_units; i++)
  {
    outputs[i] = (short_activ_t) (t_outputs[i]
            >> (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT));
  }

  // record outputs,
  recording_record(OUTPUTS,
      (void *) outputs, tcfg.num_units * sizeof (short_activ_t)
  );
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// record tick data - to be picked up by the host
// ------------------------------------------------------------------------
void record_tick_data (void)
{
  tick_record_t tick_data;

  // prepare tick data,
  tick_data.epoch   = epoch;
  tick_data.example = example_cnt;
  tick_data.event   = evt;
  tick_data.tick    = tick;

  // and record it
  recording_record(TICK_DATA, (void *) &tick_data, sizeof (tick_record_t));
}
// ------------------------------------------------------------------------
