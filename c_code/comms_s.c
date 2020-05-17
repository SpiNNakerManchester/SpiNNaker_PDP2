// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_s.h"
#include "comms_s.h"
#include "process_s.h"

// this file contains the communication routines used by S cores

// ------------------------------------------------------------------------
// process received packets (stop, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void s_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

  // check if stop packet
  if ((key & SPINN_TYPE_MASK) == SPINN_STOP_KEY)
  {
    // stop packet received
    #ifdef DEBUG
      stp_recv++;
    #endif

    // STOP decision arrived
    tick_stop = key & SPINN_STPD_MASK;

    #ifdef DEBUG_VRB
      io_printf (IO_BUF, "sc:%x\n", tick_stop);
    #endif

    // check if all other threads done
    if (sf_thrds_pend == 0)
    {
      // if done initialise semaphore
      sf_thrds_pend = 1;

      // and advance tick
      spin1_schedule_callback (sf_advance_tick, 0, 0, SPINN_S_TICK_P);
    }
    else
    {
      // if not done report processing thread done
      sf_thrds_pend -= 1;
    }

    return;
  }

  // check if network stop packet
  if ((key & SPINN_TYPE_MASK) == SPINN_STPN_KEY)
  {
    // network stop packet received
    #ifdef DEBUG
      stn_recv++;
    #endif

      // stop timer ticks,
//lap      simulation_exit ();

      // report no error,
      done(SPINN_NO_ERROR);

      // and let host know that we're ready
      simulation_ready_to_read();
      return;
  }

  // queue packet - if space available
  uint new_tail = (s_pkt_queue.tail + 1) % SPINN_SUM_PQ_LEN;
  if (new_tail == s_pkt_queue.head)
  {
      // stop timer ticks,
//lap      simulation_exit ();

      // report queue full error,
      done(SPINN_QUEUE_FULL);

      // and let host know that we're ready
      simulation_ready_to_read();
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
