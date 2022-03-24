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

#include "init_i.h"
#include "comms_i.h"
#include "process_i.h"


// ------------------------------------------------------------------------
// input core communications routines
// includes functions to transfer data between DTCM and SDRAM
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// enqueue data packet
// ------------------------------------------------------------------------
void i_receiveDataPacket (uint key, uint payload)
{
  // queue packet - if space available
  uint new_tail = (i_pkt_queue.tail + 1) % SPINN_INPUT_PQ_LEN;
  if (new_tail == i_pkt_queue.head)
  {
    // report queue full error
    stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full enqueue packet,
    i_pkt_queue.queue[i_pkt_queue.tail].key = key;
    i_pkt_queue.queue[i_pkt_queue.tail].payload = payload;
    i_pkt_queue.tail = new_tail;

    // and schedule processing thread -- if not active already
    if (!i_active)
    {
      i_active = TRUE;
      spin1_schedule_callback (i_processQueue, 0, 0, SPINN_I_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process control packet
// ------------------------------------------------------------------------
void i_receiveControlPacket (uint key, uint unused)
{
  (void) unused;

  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process tick stop packet,
  if (pkt_type == SPINN_STOP_KEY)
  {
    i_stop_packet (key);
    return;
  }

  // or process backprop sync packet,
  if (pkt_type == SPINN_SYNC_KEY)
  {
    i_sync_packet (key);
    return;
  }

  // or process network stop packet,
  if (pkt_type == SPINN_STPN_KEY)
  {
    i_net_stop_packet (key);
    return;
  }

  // or process deadlock recovery packet,
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
      i_dlrv_packet ();
    }

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
void i_processQueue (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "i_process\n");
#endif

  // access queue with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // process until queue empty,
  while (i_pkt_queue.head != i_pkt_queue.tail)
  {
    // dequeue packet,
    uint key = i_pkt_queue.queue[i_pkt_queue.head].key;
    uint payload = i_pkt_queue.queue[i_pkt_queue.head].payload;
    i_pkt_queue.head = (i_pkt_queue.head + 1) % SPINN_INPUT_PQ_LEN;

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
        if_process (key, payload);
      }
      else
      {
        // process BACKPROP phase packet
        ib_process (key, payload);
      }
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
  i_active = FALSE;

  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a tick stop packet
// ------------------------------------------------------------------------
void i_stop_packet (uint key)
{
#ifdef DEBUG
  stp_recv++;
  if (phase == SPINN_BACKPROP) wrng_fph++;
  uint tick_recv = key & SPINN_TICK_MASK;
  if (tick_recv != tick) wrng_pth++;
#endif

  // get tick STOP decision,
  //NOTE: be careful with variable size
  tick_stop = (key & SPINN_STOP_MASK) ? 1 : 0;

  // and advance tick
  spin1_schedule_callback (if_advance_tick, 0, 0, SPINN_I_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a sync packet
// ------------------------------------------------------------------------
void i_sync_packet (uint key)
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
  spin1_schedule_callback (ib_advance_tick, 0, 0, SPINN_I_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop packet
// ------------------------------------------------------------------------
void i_net_stop_packet (uint key)
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

    // and restore interrupts after flag access
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a deadlock recovery packet
// ------------------------------------------------------------------------
void i_dlrv_packet (void)
{
  // prepare to restart tick
  spin1_schedule_callback (tick_init, SPINN_RESTART, 0, SPINN_I_TICK_P);
}
// ------------------------------------------------------------------------
