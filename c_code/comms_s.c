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

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_s.h"
#include "comms_s.h"
#include "process_s.h"


// ------------------------------------------------------------------------
// sum core communications routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// enqueue data packet
// ------------------------------------------------------------------------
void s_receiveDataPacket (uint key, uint payload)
{
  // queue packet - if space available
  uint new_tail = (s_pkt_queue.tail + 1) % SPINN_SUM_PQ_LEN;
  if (new_tail == s_pkt_queue.head)
  {
      // report queue full error
      stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full enqueue packet,
    s_pkt_queue.queue[s_pkt_queue.tail].key = key;
    s_pkt_queue.queue[s_pkt_queue.tail].payload = payload;
    s_pkt_queue.tail = new_tail;

    // and schedule processing thread -- if not active already
    if (!s_active)
    {
      s_active = TRUE;
      spin1_schedule_callback (s_processQueue, 0, 0, SPINN_S_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process control packet
// ------------------------------------------------------------------------
void s_receiveControlPacket (uint key, uint unused)
{
  (void) unused;

  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process forward sync generation packet,
  if (pkt_type == SPINN_FSGN_KEY)
  {
    s_fsgn_packet ();
    return;
  }

  // process backprop sync generation packet,
  if (pkt_type == SPINN_BSGN_KEY)
  {
    s_bsgn_packet ();
    return;
  }

  // process tick stop packet,
  if (pkt_type == SPINN_STOP_KEY)
  {
    s_stop_packet (key);
    return;
  }

  // process backprop sync packet,
  if (pkt_type == SPINN_SYNC_KEY)
  {
    s_sync_packet (key);
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
      s_dlrv_packet ();
    }

    return;
  }

  // process network stop packet,
  if (pkt_type == SPINN_STPN_KEY)
  {
    s_net_stop_packet (key);
    return;
  }

#ifdef DEBUG
  // or report unexpected packet type
  stage_done (SPINN_UNXPD_PKT, key);
#endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process packet queue until empty
// ------------------------------------------------------------------------
void s_processQueue (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "s_process\n");
#endif

  // access queue with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // process until queue empty,
  while (s_pkt_queue.head != s_pkt_queue.tail)
  {
    // if not empty dequeue packet,
    uint key = s_pkt_queue.queue[s_pkt_queue.head].key;
    uint payload = s_pkt_queue.queue[s_pkt_queue.head].payload;
    s_pkt_queue.head = (s_pkt_queue.head + 1) % SPINN_SUM_PQ_LEN;

    // restore interrupts after queue access,
    spin1_mode_restore (cpsr);

    uint pkt_type = key & SPINN_TYPE_MASK;

    // process data packet,
    if (pkt_type == SPINN_DATA_KEY)
    {
      // check packet phase and process accordingly
      uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

      if (ph == SPINN_FORWARD)
      {
        // process FORWARD phase packet
        sf_process (key, payload);
      }
      else
      {
        // process BACKPROP phase packet
        sb_process (key, payload);
      }
    }

    // process LDS packet,
    else if (pkt_type == SPINN_LDSA_KEY)
    {
      s_lds_packet (payload);
    }

#ifdef DEBUG
    // or report unknown packet type,
    else
    {
      stage_done (SPINN_UNXPD_PKT, key);
    }
#endif

    // and access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // flag going to sleep,
  s_active = FALSE;

  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process LDS packet: accumulate the received partial link delta sums
// ------------------------------------------------------------------------
void s_lds_packet (uint payload)
{
#ifdef DEBUG
  lds_recv++;
#endif

  // add the received value to the total so far,
  s_lds_part += (lds_t) payload;

  // increment the count of partial link delta sums arrived,
  s_lds_arrived++;

  // check whether all the partial sums have arrived
  if (s_lds_arrived == scfg.lds_expected)
  {
    // broadcast (first subgroup) or relay (all others) lds value
    while (!spin1_send_mc_packet (ldsKey, s_lds_part, WITH_PAYLOAD));

#ifdef DEBUG
    lds_sent++;
#endif

    // access thread semaphore with interrupts disabled
    uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(sb_thrds_pend & SPINN_THRD_LDSA)) wrng_cth++;
#endif

    // check if all other threads done
    if (sb_thrds_pend == SPINN_THRD_LDSA)
    {
#ifdef DEBUG
      // report processing thread done,
      sb_thrds_pend = 0;
#endif

      // restore interrupts after flag access,
      spin1_mode_restore (cpsr);

      // report backprop sync generation ready
      if (scfg.is_tree_root)
      {
        while (!spin1_send_mc_packet (bpsKey, 0, NO_PAYLOAD));

#ifdef DEBUG
        bsg_sent++;
#endif
      }
    }
    else
    {
      // report processing thread done,
      sb_thrds_pend &= ~SPINN_THRD_LDSA;

      // and restore interrupts after flag access
      spin1_mode_restore (cpsr);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a tick stop packet
// ------------------------------------------------------------------------
void s_stop_packet (uint key)
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
  spin1_schedule_callback (sf_advance_tick, 0, 0, SPINN_S_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a backprop sync packet
// ------------------------------------------------------------------------
void s_sync_packet (uint key)
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
  spin1_schedule_callback (sb_advance_tick, 0, 0, SPINN_S_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop packet
// ------------------------------------------------------------------------
void s_net_stop_packet (uint key)
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
      //TODO: check if need to schedule or can simply call
      spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
    }
  }
  else
  {
    // flag ready for net_stop decision,
    net_stop_rdy = TRUE;

    // and restore interrupts after flag access,
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a backprop sync generation packet
// ------------------------------------------------------------------------
void s_bsgn_packet (void)
{
#ifdef DEBUG
  bsg_recv++;
  if (phase == SPINN_FORWARD) wrng_bph++;
#endif

  // update count of sync packets,
  s_bsgn_arrived++;

  // and check if all expected packets arrived
  if (s_bsgn_arrived == scfg.bsgn_expected)
  {
    // check if all other threads done
    if (sb_thrds_pend == SPINN_THRD_BSGN)
    {
#ifdef DEBUG
      // report processing thread done,
      sb_thrds_pend = 0;
#endif

      // report backprop sync generation ready
      while (!spin1_trigger_user_event (bpsKey, 0));

#ifdef DEBUG
      bsg_sent++;
#endif
    }
    else
    {
      // report sync thread done
      sb_thrds_pend &= ~SPINN_THRD_BSGN;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a forward sync generation packet
// ------------------------------------------------------------------------
void s_fsgn_packet (void)
{
#ifdef DEBUG
  fsg_recv++;
  if (phase == SPINN_BACKPROP) wrng_fph++;
#endif

  // update count of forward sync generation packets,
  s_fsgn_arrived++;

  // and check if all expected packets arrived
  if (s_fsgn_arrived == scfg.fsgn_expected)
  {
    // report forward sync generation ready
    while (!spin1_trigger_user_event (fsgKey, 0));

#ifdef DEBUG
    fsg_sent++;
#endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a deadlock recovery packet
// ------------------------------------------------------------------------
void s_dlrv_packet (void)
{
  // restart tick
  spin1_schedule_callback (tick_init, SPINN_RESTART, 0, SPINN_S_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// send a control packet - used in FIQ callbacks
// ------------------------------------------------------------------------
void s_sendControlPacket (uint key, uint unused)
{
  (void) unused;

  // send control packet - no payload
  while (!spin1_send_mc_packet(key, 0, NO_PAYLOAD));
}
// ------------------------------------------------------------------------
