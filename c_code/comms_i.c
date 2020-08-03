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
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process received packets (stop, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void i_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

  uint pkt_type = key & SPINN_TYPE_MASK;

  // check if stop packet
  if (pkt_type == SPINN_STOP_KEY)
  {
    // stop packet received
#ifdef DEBUG
    stp_recv++;
#endif

    // tick STOP decision arrived
    tick_stop = key & SPINN_STPD_MASK;

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(if_thrds_pend & SPINN_THRD_STOP))
      wrng_sth++;
#endif

    // check if all other threads done
    if (if_thrds_pend == SPINN_THRD_STOP)
    {
      // if done initialise semaphore,
      if_thrds_pend = SPINN_IF_THRDS;

      // and advance tick
      spin1_schedule_callback (if_advance_tick, 0, 0, SPINN_I_TICK_P);
    }
    else
    {
      // if not done report processing thread done
      if_thrds_pend &= ~SPINN_THRD_STOP;
    }

    return;
  }

  // check if network stop packet
  if (pkt_type == SPINN_STPN_KEY)
  {
    // network stop packet received
#ifdef DEBUG
    stn_recv++;
#endif

    // network stop decision arrived
    net_stop = key & SPINN_STPD_MASK;

    // check if ready for network stop decision
    if (net_stop_rdy)
    {
      // clear flag,
      net_stop_rdy = FALSE;

      // and decide what to do
      if (net_stop)
      {
        // finish stage and report no error
        spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
      }
    }
    else
    {
      // flag ready for net_stop decision
      net_stop_rdy = TRUE;
    }

    return;
  }

  // queue packet - if space available
  uint new_tail = (i_pkt_queue.tail + 1) % SPINN_INPUT_PQ_LEN;
  if (new_tail == i_pkt_queue.head)
  {
    // report queue full error
    stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full queue packet,
    i_pkt_queue.queue[i_pkt_queue.tail].key = key;
    i_pkt_queue.queue[i_pkt_queue.tail].payload = payload;
    i_pkt_queue.tail = new_tail;

    // and schedule processing thread -- if not active already
    if (!i_active)
    {
      i_active = TRUE;
      spin1_schedule_callback (i_process, 0, 0, SPINN_I_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------
