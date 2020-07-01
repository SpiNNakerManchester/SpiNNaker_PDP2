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
    if (if_thrds_pend == 0)
    {
      // if done initialise semaphore,
      if_thrds_pend = 1;

      // and advance tick
      spin1_schedule_callback (if_advance_tick, 0, 0, SPINN_I_TICK_P);
    }
    else
    {
      // if not done report processing thread done
      if_thrds_pend -= 1;
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

    // report no error
    stage_done (SPINN_NO_ERROR);
    return;
  }

  // queue packet - if space available
  uint new_tail = (i_pkt_queue.tail + 1) % SPINN_INPUT_PQ_LEN;
  if (new_tail == i_pkt_queue.head)
  {
    // report queue full error
    stage_done (SPINN_QUEUE_FULL);
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
