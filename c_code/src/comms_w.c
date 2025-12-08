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

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_w.h"
#include "comms_w.h"
#include "process_w.h"


// ------------------------------------------------------------------------
// weight core communications routines
// includes functions to transfer data between DTCM and SDRAM
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process data packet
// ------------------------------------------------------------------------
void w_receiveDataPacket (uint key, uint payload)
{
  // check packet phase,
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // FORWARD-phase packets are handled immediately
  if (ph == SPINN_FORWARD)
  {
    w_handleFWDPacket (key, payload);
    return;
  }

  // BACKPROP-phase packets are queued for background processing
  uint new_tail = (w_pkt_queue.tail + 1) % SPINN_WEIGHT_PQ_LEN;

  // check if space in packet queue,
  if (new_tail == w_pkt_queue.head)
  {
    // report queue full error
    stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full enqueue packet,
    w_pkt_queue.queue[w_pkt_queue.tail].key = key;
    w_pkt_queue.queue[w_pkt_queue.tail].payload = payload;
    w_pkt_queue.tail = new_tail;

    // and schedule BACKPROP processing thread
    if (!wb_active && (phase == SPINN_BACKPROP))
    {
      wb_active = TRUE;
      spin1_schedule_callback (w_processBKPQueue, 0, 0, SPINN_WB_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process control packet
// ------------------------------------------------------------------------
void w_receiveControlPacket (uint key, uint unused)
{
  (void) unused;

  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process tick stop packet,
  if (pkt_type == SPINN_STOP_KEY)
  {
    w_stop_packet (key);
    return;
  }

  // or process backprop sync packet,
  if (pkt_type == SPINN_SYNC_KEY)
  {
    w_sync_packet (key);
    return;
  }

  // or process network stop packet,
  if (pkt_type == SPINN_STPN_KEY)
  {
    w_net_stop_packet (key);
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
      w_dlrv_packet ();
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
// handle FORWARD-phase packets
// (FORWARD, stop, net_stop and sync types)
// ------------------------------------------------------------------------
void w_handleFWDPacket (uint key, uint payload)
{
  // check packet type,
  uint pkt_type = key & SPINN_TYPE_MASK;

  // process FORWARD data packet,
  if (pkt_type == SPINN_DATA_KEY)
  {
    w_forward_packet (key, payload);
    return;
  }

#ifdef DEBUG
  // or report unexpected packet type
  stage_done (SPINN_UNXPD_PKT, key);
#endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP-phase packet queue until empty
// ------------------------------------------------------------------------
void w_processBKPQueue (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "w_processBKPQueue\n");
#endif

  // access queue with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // process until queue empty,
  while (w_pkt_queue.head != w_pkt_queue.tail)
  {
    // dequeue packet,
    uint key = w_pkt_queue.queue[w_pkt_queue.head].key;
    uint payload = w_pkt_queue.queue[w_pkt_queue.head].payload;
    w_pkt_queue.head = (w_pkt_queue.head + 1) % SPINN_WEIGHT_PQ_LEN;

    // restore interrupts after queue access,
    spin1_mode_restore (cpsr);

    // check packet type,
    uint pkt_type = key & SPINN_TYPE_MASK;

    // process BACKPROP data packet,
    if (pkt_type == SPINN_DATA_KEY)
    {
      wb_process (key, payload);
    }

    // or process LDS result packet,
    else if (pkt_type == SPINN_LDSA_KEY)
    {
      w_ldsa_packet (payload);
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
  wb_active = FALSE;

  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a FORWARD data packet
// ------------------------------------------------------------------------
void w_forward_packet (uint key, uint payload)
{
#ifdef DEBUG
  recv_fwd++;
  if (phase == SPINN_BACKPROP) wrng_fph++;
#endif

  // get output index: mask out phase, core and block data,
  uint inx = key & SPINN_OUTPUT_MASK;

  // store received unit output,
  w_outputs[wf_comms][inx] = (activation_t) payload;

  // store output for use in BACKPROP phase,
  store_output (inx);

  // update scoreboard,
  wf_arrived++;

  // and check if all expected unit outputs have arrived
  if (wf_arrived == wcfg.num_rows)
  {
    // trigger forward sync generation,
    while (!spin1_trigger_user_event (fsgKey, 0));

#ifdef DEBUG
    fsg_sent++;
#endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process an LDS result packet
// ------------------------------------------------------------------------
void w_ldsa_packet (uint payload)
{
#ifdef DEBUG
  lds_recv++;
#endif

  //TODO: need to synchronise the arrival of final ldsa packet
  // to all w cores!

  // the final link delta sum for the epoch arrived
  w_lds_final = (lds_t) payload;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a tick stop packet
// ------------------------------------------------------------------------
void w_stop_packet (uint key)
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
  spin1_schedule_callback (wf_advance_tick, 0, 0, SPINN_W_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a backprop sync packet
// ------------------------------------------------------------------------
void w_sync_packet (uint key)
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
  spin1_schedule_callback (wb_advance_tick, 0, 0, SPINN_W_TICK_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop packet
// ------------------------------------------------------------------------
void w_net_stop_packet (uint key)
{
#ifdef DEBUG
  stn_recv++;
#endif

  // network stop decision arrived
  net_stop = key & SPINN_STPD_MASK;

  // check if ready for network stop decision
  if (epoch_rdy)
  {
    // clear flag for next tick,
    epoch_rdy = FALSE;

    // and decide what to do
    if (net_stop)
    {
      // finish stage and report no error
      //TODO: check if need to schedule or can simply call
      spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
    }
    else
    {
      // trigger computation
      spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
    }
  }
  else
  {
    // flag as ready
    net_stop_rdy = TRUE;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a deadlock recovery packet
// ------------------------------------------------------------------------
void w_dlrv_packet (void)
{
  // prepare to restart tick,
  spin1_schedule_callback (tick_init, SPINN_RESTART, 0, SPINN_W_TICK_P);

  // and trigger computation
  if (phase == SPINN_FORWARD)
  {
    spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores unit output of the specified unit for the current tick
// ------------------------------------------------------------------------
void store_output (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "store_output\n");
#endif

  w_output_history[(tick * wcfg.num_rows) + inx] = w_outputs[wf_comms][inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores all unit outputs for the requested tick
// ------------------------------------------------------------------------
void restore_outputs (uint tick)
{
#ifdef TRACE
  io_printf (IO_BUF, "restore_outputs\n");
#endif

  for (uint inx = 0; inx < wcfg.num_rows; inx++)
  {
    w_outputs[0][inx] = w_output_history[(tick * wcfg.num_rows) + inx];
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// send a control packet - used in FIQ callbacks
// ------------------------------------------------------------------------
void w_sendControlPacket (uint key, uint unused)
{
  (void) unused;

  // send control packet - no payload
  while (!spin1_send_mc_packet(key, 0, NO_PAYLOAD));
}
// ------------------------------------------------------------------------
