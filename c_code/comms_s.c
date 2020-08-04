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
// enqueue received packet (FORWARD, BACKPROP, ldsa, ldst, stop and stpn types)
// ------------------------------------------------------------------------
void s_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

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
      spin1_schedule_callback (s_processQueue, 0, 0, SPINN_S_PROCESS_P);
    }
  }
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

  // access queue with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // process until queue empty
  while (s_pkt_queue.head != s_pkt_queue.tail)
  {
    // if not empty dequeue packet,
    uint key = s_pkt_queue.queue[s_pkt_queue.head].key;
    uint payload = s_pkt_queue.queue[s_pkt_queue.head].payload;
    s_pkt_queue.head = (s_pkt_queue.head + 1) % SPINN_SUM_PQ_LEN;

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
        s_forward_packet (key, payload);
      }
      else
      {
        // process BACKPROP phase packet
        s_backprop_packet (key, payload);
      }
    }

    // check for an LDS "accumulation" packet,
    else if (pkt_type == SPINN_LDSA_KEY)
    {
      // process LDS "accumulation" packet
      s_ldsa_packet (payload);
    }

    // check for LDS "total" packet,
    else if (pkt_type == SPINN_LDST_KEY)
    {
      // process LDS "total" packet
      s_ldst_packet (payload);
    }

    // check if stop packet,
    else if (pkt_type == SPINN_STOP_KEY)
    {
      // stop packet received
      s_stop_packet (key);
    }

    // check if network stop packet,
    else if (pkt_type == SPINN_STPN_KEY)
    {
      // network stop packet received
      s_net_stop_packet (key);
    }

    // and access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // when done, flag that going to sleep,
  s_active = FALSE;

  // restore interrupts and leave
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------
