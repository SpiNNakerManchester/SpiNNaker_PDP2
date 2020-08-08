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
// enqueue received packet
// (FORWARD, BACKPROP, stop and net_stop types)
// ------------------------------------------------------------------------
void i_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

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

    // check if data packet,
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

    // check if stop packet,
    else if (pkt_type == SPINN_STOP_KEY)
    {
      // stop packet received
      i_stop_packet (key);
    }

    // check if network stop packet,
    else if (pkt_type == SPINN_STPN_KEY)
    {
      // network stop packet received
      i_net_stop_packet (key);
    }

#ifdef DEBUG
    // report unknown packet type,
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
#endif

  // tick STOP decision arrived,
  tick_stop = key & SPINN_STPD_MASK;

  // access thread semaphore with interrupts disabled,
  uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(if_thrds_pend & SPINN_THRD_STOP))
      wrng_sth++;
#endif

  // and check if all other threads done
  if (if_thrds_pend == SPINN_THRD_STOP)
  {
    // if done initialise semaphore,
    if_thrds_pend = SPINN_IF_THRDS;

    // restore interrupts after semaphore access,
    spin1_mode_restore (cpsr);

    // and advance tick
    if_advance_tick ();
  }
  else
  {
    // if not done report processing thread done
    if_thrds_pend &= ~SPINN_THRD_STOP;

    // and restore interrupts after semaphore access
    spin1_mode_restore (cpsr);
  }
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
// stores unit net received for the current tick
// ------------------------------------------------------------------------
void store_net (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "store_nets\n");
#endif

  i_net_history[(tick * icfg.num_units) + inx] = i_nets[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores unit net for the requested tick
// ------------------------------------------------------------------------
void restore_net (uint inx, uint tick)
{
#ifdef TRACE
  io_printf (IO_BUF, "restore_nets\n");
#endif

  i_nets[inx] = i_net_history[(tick * icfg.num_units) + inx];
}
// ------------------------------------------------------------------------
