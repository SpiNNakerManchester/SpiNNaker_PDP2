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
// process received packets (stop, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void s_receivePacket (uint key, uint payload)
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

    // STOP decision arrived
    tick_stop = key & SPINN_STPD_MASK;

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(sf_thrds_pend & SPINN_THRD_STOP))
      wrng_sth++;
#endif

    // check if all other threads done
    if (sf_thrds_pend == SPINN_THRD_STOP)
    {
      // if done initialise semaphore
      sf_thrds_pend = SPINN_SF_THRDS;

      // and advance tick
      spin1_schedule_callback (sf_advance_tick, 0, 0, SPINN_S_TICK_P);
    }
    else
    {
      // if not done report processing thread done
      sf_thrds_pend &= ~SPINN_THRD_STOP;
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
  uint new_tail = (s_pkt_queue.tail + 1) % SPINN_SUM_PQ_LEN;
  if (new_tail == s_pkt_queue.head)
  {
      // report queue full error
      stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full queue packet,
    s_pkt_queue.queue[s_pkt_queue.tail].key = key;
    s_pkt_queue.queue[s_pkt_queue.tail].payload = payload;
    s_pkt_queue.tail = new_tail;

    // and schedule processing thread -- if not active already
    if (!s_active)
    {
      s_active = TRUE;
      spin1_schedule_callback (s_process, 0, 0, SPINN_S_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------
