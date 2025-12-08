/*
 * Copyright (c) 2015 The University of Manchester
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
// process data packet
// ------------------------------------------------------------------------
void t_receiveDataPacket (uint key, uint payload)
{
  // check packet phase,
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // BACKPROP-phase packets are handled immediately
  if (ph == SPINN_BACKPROP)
  {
    t_handleBKPPacket (key, payload);
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
// process control packet
// ------------------------------------------------------------------------
void t_receiveControlPacket (uint key, uint unused)
{
  (void) unused;

  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process forward sync gen packet,
  if (pkt_type == SPINN_FSGN_KEY)
  {
    t_fsgn_packet ();
    return;
  }

  // process backprop sync generation packet,
  if (pkt_type == SPINN_BSGN_KEY)
  {
    t_bsgn_packet ();
    return;
  }

  // process criterion packet,
  if (pkt_type == SPINN_CRIT_KEY)
  {
    t_criterion_packet (key);
    return;
  }

  // process tick stop packet,
  if (pkt_type == SPINN_STOP_KEY)
  {
    t_stop_packet (key);
    return;
  }

  // process backprop sync packet,
  if (pkt_type == SPINN_SYNC_KEY)
  {
    t_sync_packet (key);
    return;
  }

  // process deadlock recovery packet,
  if (pkt_type == SPINN_DLRV_KEY)
  {
#ifdef DEBUG
    dlr_recv++;
#endif

    if (key & SPINN_ABRT_MASK)
    {
      // report timeout error
      stage_done (SPINN_TIMEOUT_EXIT, 0);
    }
    else
    {
      t_dlrv_packet ();
    }

    return;
  }

  // process network stop packet,
  if (pkt_type == SPINN_STPN_KEY)
  {
    t_net_stop_packet (key);
    return;
  }

#ifdef DEBUG
  // or report unexpected packet type
  stage_done (SPINN_UNXPD_PKT, key);
#endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// handle BACKPROP-phase packets
// (BACKPROP type)
// ------------------------------------------------------------------------
void t_handleBKPPacket (uint key, uint payload)
{
  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process BACKPROP data packet,
  if (pkt_type == SPINN_DATA_KEY)
  {
    t_backprop_packet (key, payload);
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
  //NOTE: be careful with variable size
  uchar crit_recv = (key & SPINN_CRIT_MASK) ? 1 : 0;
  
  tf_crit_prev = tf_crit_prev && crit_recv;

  // update scoreboard,
  tf_crit_arrived++;

  // and check if all criterion packets arrived
  if (tf_crit_arrived == tcfg.crit_expected)
  {
    // check if all other threads are done
    if (tf_thrds_pend == SPINN_THRD_CRIT)
    {
#ifdef DEBUG
      // report thread done
      tf_thrds_pend = 0;
#endif

      // send criterion/stop packet,
      send_stop_crit ();

      // and advance tick if last_output_group
      //NOTE: last output group does not get a tick stop packet
      // so it's ready to advance tick
      if (tcfg.is_last_output)
      {
        spin1_schedule_callback (tf_advance_tick, 0, 0, SPINN_T_TICK_P);
      }
    }
    else
    {
      // report thread done
      tf_thrds_pend &= ~SPINN_THRD_CRIT;
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

  // check if all other threads are done
  if (tf_thrds_pend == SPINN_THRD_FSGN)
  {
#ifdef DEBUG
    // report thread done
    tf_thrds_pend = 0;
#endif

    // send criterion/stop packet,
    send_stop_crit();

    // and advance tick if last_output_group
    //NOTE: last output group does not get a tick stop packet
    // so it's ready to advance tick
    if (tcfg.is_last_output)
    {
      spin1_schedule_callback (tf_advance_tick, 0, 0, SPINN_T_TICK_P);
    }
  }
  else
  {
    // report thread done
    tf_thrds_pend &= ~SPINN_THRD_FSGN;
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
  if (phase == SPINN_BACKPROP) wrng_fph++;
  uint tick_recv = key & SPINN_TICK_MASK;
  if (tick_recv != tick) wrng_pth++;
#endif

  // get tick stop decision,
  //NOTE: be careful with variable size
  tick_stop = (key & SPINN_STOP_MASK) ? 1 : 0;

  // and advance tick
  spin1_schedule_callback (tf_advance_tick, 0, 0, SPINN_T_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a backprop sync packet
// ------------------------------------------------------------------------
void t_sync_packet (uint key)
{
#ifdef DEBUG
  spk_recv++;
  if (phase == SPINN_FORWARD) wrng_bph++;
  uint tick_recv = key & SPINN_TICK_MASK;
  if (tick_recv != tick) wrng_pth++;
#else
  (void) key;
#endif

  // advance tick
  spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_T_TICK_P);
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
  // prepare to restart tick,
  spin1_schedule_callback (tick_init, SPINN_RESTART, 0, SPINN_T_TICK_P);

  // and trigger computation
  if (phase == SPINN_BACKPROP)
  {
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
#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(tb_thrds_pend & SPINN_THRD_COMS))
      wrng_cth++;
#endif

    // and check if all other threads are done,
    if (tb_thrds_pend == SPINN_THRD_COMS)
    {
#ifdef DEBUG
      // report comms thread done
      tb_thrds_pend = 0;
#endif

      // send backprop sync packet to allow next tick to start,
      send_sync ();

#ifdef DEBUG
      if (tcfg.is_last_output) spk_sent++;
      else bsg_sent++;
#endif

      // and advance tick
      //NOTE: last t core does *not* get a sync packet
      if (tcfg.is_last_output)
      {
        spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_T_TICK_P);
      }
    }
    else
    {
      // report comms thread done
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
  {
    wrng_bph++;
  }
#endif

  // update scoreboard,
  tb_bsgn_arrived++;

  // and check if all backprop sync gen packets arrived
  if (tb_bsgn_arrived == tb_bsgn_expected)
  {
    // check if all other threads done
    if (tb_thrds_pend == SPINN_THRD_BSGN)
    {
#ifdef DEBUG
      // report sync thread done
      tb_thrds_pend = 0;
#endif

      // send backprop sync packet to allow next tick to start,
      send_sync ();

#ifdef DEBUG
      if (tcfg.is_last_output)
      {
        spk_sent++;
      }
      else
      {
        bsg_sent++;
      }
#endif

      // and advance tick
      //NOTE: last t core does *not* get a sync packet
      if (tcfg.is_last_output)
      {
        spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_T_TICK_P);
      }
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
// send a control packet - used in FIQ callbacks
// ------------------------------------------------------------------------
void t_sendControlPacket (uint key, uint unused)
{
  (void) unused;

  // send control packet - no payload
  while (!spin1_send_mc_packet(key, 0, NO_PAYLOAD));
}
// ------------------------------------------------------------------------
