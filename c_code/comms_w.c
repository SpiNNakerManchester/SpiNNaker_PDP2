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
// initial handling of received packets
// (FORWARD, BACKPROP, ldsr, stop, net_stop and sync types)
// ------------------------------------------------------------------------
void w_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

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

  // or process tick stop packet,
  if (pkt_type == SPINN_STOP_KEY)
  {
    w_stop_packet (key);
    return;
  }

  // or process network stop packet,
  if (pkt_type == SPINN_STPN_KEY)
  {
    w_net_stop_packet (key);
    return;
  }

  // or process synchronisation packet,
  if (pkt_type == SPINN_SYNC_KEY)
  {
    w_sync_packet ();
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

    // process LDS result packet,
    else if (pkt_type == SPINN_LDSR_KEY)
    {
      w_ldsr_packet (payload);
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
  if (phase == SPINN_BACKPROP)
    wrng_fph++;

  uint blk = (key & SPINN_BLOCK_MASK) >> SPINN_BLOCK_SHIFT;
  if (blk != wcfg.row_blk)
  {
    pkt_fwbk++;
    return;
  }
#endif

  // get output index: mask out phase, core and block data,
  uint inx = key & SPINN_BLKOUT_MASK;

  // store received unit output,
  w_outputs[wf_comms][inx] = (activation_t) payload;

  // store output for use in BACKPROP phase,
  store_output (inx);

  // update scoreboard,
  wf_arrived++;

  // and check if all expected unit outputs have arrived
  if (wf_arrived == wcfg.num_rows)
  {
    // initialise scoreboard for next tick,
    wf_arrived = 0;

    // update pointer to received unit outputs,
    wf_comms = 1 - wf_comms;

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(wf_thrds_pend & SPINN_THRD_COMS))
      wrng_cth++;
#endif

    // and check if all other threads are done,
    if (wf_thrds_pend == SPINN_THRD_COMS)
    {
      // if done initialise thread semaphore,
      wf_thrds_pend = SPINN_WF_THRDS;

      // and advance tick
      spin1_schedule_callback (wf_advance_tick, 0, 0, SPINN_WF_TICK_P);
    }
    else
    {
      // if not done report comms thread done
      wf_thrds_pend &= ~SPINN_THRD_COMS;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a tick stop packet
// ------------------------------------------------------------------------
void w_stop_packet (uint key)
{
#ifdef DEBUG
  stp_recv++;
  if (phase == SPINN_BACKPROP)
    wrng_fph++;
#endif

  // tick stop decision arrived,
  tick_stop = key & SPINN_STPD_MASK;

#if defined(DEBUG) && defined(DEBUG_THRDS)
  if (!(wf_thrds_pend & SPINN_THRD_STOP))
    wrng_sth++;
#endif

  // check if all other threads done
  if (wf_thrds_pend == SPINN_THRD_STOP)
  {
    // if done initialise thread semaphore,
    wf_thrds_pend = SPINN_WF_THRDS;

    // and advance tick
    spin1_schedule_callback (wf_advance_tick, 0, 0, SPINN_WF_TICK_P);
  }
  else
  {
    // if not done report stop thread done
    wf_thrds_pend &= ~SPINN_THRD_STOP;
  }
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
  if (sync_rdy && epoch_rdy)
  {
    // clear flags for next tick,
    sync_rdy = FALSE;
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
// process a sync packet
// ------------------------------------------------------------------------
void w_sync_packet (void)
{
#ifdef DEBUG
  spk_recv++;
#endif

  // update count of sync packets,
  w_sync_arrived++;

  // and check if all expected packets arrived
  if (w_sync_arrived == wcfg.sync_expected)
  {
    // prepare for next synchronisation,
    w_sync_arrived = 0;

    // and check if can trigger next example computation
    if (net_stop_rdy && epoch_rdy)
    {
      // clear flags for next tick,
      net_stop_rdy = FALSE;
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
        // and trigger computation
        spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
      }
    }
    else
    {
      // flag as ready
      sync_rdy = TRUE;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process an LDS result packet
// ------------------------------------------------------------------------
void w_ldsr_packet (uint payload)
{
#ifdef DEBUG
  ldr_recv++;
#endif

  // the final link delta sum for the epoch arrived
  w_lds_final = (lds_t) payload;

  // access thread semaphore with interrupts disabled,
  uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
  if (!(wb_thrds_pend & SPINN_THRD_LDSR))
    wrng_cth++;
#endif

  // check if all other threads done
  if (wb_thrds_pend == SPINN_THRD_LDSR)
  {
    // initialise semaphore (no link delta summation in next tick),
    wb_thrds_pend = SPINN_WB_THRDS;

    // restore interrupts after semaphore access,
    spin1_mode_restore (cpsr);

    // and advance tick
    wb_advance_tick ();
  }
  else
  {
    // if not done report processing thread done,
    wb_thrds_pend &= ~SPINN_THRD_LDSR;

    // and restore interrupts after semaphore access
    spin1_mode_restore (cpsr);
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
