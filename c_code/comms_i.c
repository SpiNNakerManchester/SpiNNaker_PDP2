// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"

#include "comms_i.h"
#include "process_i.h"

// this files contains the communication routines used by I cores

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint coreID;               // 5-bit virtual core ID
extern uint coreKey;              // 21-bit core packet ID
extern uint bkpKey;               // 32-bit packet ID for BACKPROP phase
extern uint stpKey;               // 32-bit packet ID for stop criterion

extern uint         epoch;        // current training iteration
extern uint         example;      // current example in epoch
extern uint         evt;          // current event in example
extern uint         num_ticks;    // number of ticks in current event
extern proc_phase_t phase;        // FORWARD or BACKPROP
extern uint         tick;         // current tick in phase
extern uchar        tick_stop;    // current tick stop decision

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern global_conf_t  mlpc;       // network-wide configuration parameters
extern chip_struct_t  ccfg;       // chip configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// input core variables
// ------------------------------------------------------------------------
extern pkt_queue_t      i_pkt_queue;   // queue to hold received b-d-ps
extern uchar            i_active;      // processing b-d-ps from queue?
extern uint             if_thrds_done; // sync. semaphore: proc & stop
//#extern uint             ib_thrds_done; // sync. semaphore: proc & stop
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
#ifdef DEBUG
  extern uint pkt_sent;  // total packets sent
  extern uint sent_bkp;  // packets sent in BACKPROP phase
  extern uint pkt_recv;  // total packets received
  extern uint recv_fwd;  // packets received in FORWARD phase
  extern uint recv_bkp;  // packets received in BACKPROP phase
  extern uint spk_sent;  // sync packets sent
  extern uint spk_recv;  // sync packets received
  extern uint stp_sent;  // stop packets sent
  extern uint stp_recv;  // stop packets received
  extern uint wrng_phs;  // packets received in wrong phase
  extern uint wrng_tck;  // FORWARD packets received in wrong tick
  extern uint wrng_btk;  // BACKPROP packets received in wrong tick
#endif
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process received packets (stop, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void i_receivePacket (uint key, uint payload)
{
  // check if stop packet
  if ((key & SPINN_STOP_MASK) == SPINN_STPR_KEY)
  {
    // sync packet received
    #ifdef DEBUG
      stp_recv++;
    #endif

    // STOP decision arrived
    tick_stop = (key & SPINN_STPD_MASK) >> SPINN_STPD_SHIFT;

    #ifdef DEBUG_VRB
      io_printf (IO_BUF, "sc:%x\n", tick_stop);
    #endif

    // check if all threads done
    if (if_thrds_done == 0)
    {
      // if done initialize semaphore,
      if_thrds_done = 1;

      // and advance tick
      spin1_schedule_callback (if_advance_tick, NULL, NULL, SPINN_I_TICK_P);
    }
    else
    {
      // if not done report processing thread done
      if_thrds_done -= 1;
    }

    return;
  }

  #ifdef DEBUG
    pkt_recv++;
  #endif

  // check if space in packet queue,
  uint new_tail = (i_pkt_queue.tail + 1) % SPINN_INPUT_PQ_LEN;

  if (new_tail == i_pkt_queue.head)
  {
    // if queue full exit and report failure
    spin1_exit (SPINN_QUEUE_FULL);
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
      spin1_schedule_callback (i_process, NULL, NULL, SPINN_I_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------
