// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_s.h"
#include "process_s.h"

// this files contains the communication routines used by S cores

// ------------------------------------------------------------------------
// process received packets (stop, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void s_receivePacket (uint key, uint payload)
{
  // check if stop packet
  if ((key & SPINN_STOP_MASK) == SPINN_STPR_KEY)
  {
    // stop packet received
    #ifdef DEBUG
      stp_recv++;
    #endif

    // STOP decision arrived
    tick_stop = (key & SPINN_STPD_MASK) >> SPINN_STPD_SHIFT;

    #ifdef DEBUG_VRB
      io_printf (IO_BUF, "sc:%x\n", tick_stop);
    #endif

    // check if all threads done
    if (sf_thrds_pend == 0)
    {
      // if done initialize semaphore
      sf_thrds_pend = 1;

      // and advance tick
      spin1_schedule_callback (sf_advance_tick, NULL, NULL, SPINN_S_TICK_P);
    }
    else
    {
      // if not done report processing thread done
      sf_thrds_pend -= 1;
    }
  }
  else
  {
    #ifdef DEBUG
      pkt_recv++;
    #endif

    // check if space in packet queue,
    uint new_tail = (s_pkt_queue.tail + 1) % SPINN_SUM_PQ_LEN;

    if (new_tail == s_pkt_queue.head)
    {
      // if queue full exit and report failure
      spin1_exit (SPINN_QUEUE_FULL);
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
        spin1_schedule_callback (s_process, NULL, NULL, SPINN_S_PROCESS_P);
      }
    }
  }
}
// ------------------------------------------------------------------------
