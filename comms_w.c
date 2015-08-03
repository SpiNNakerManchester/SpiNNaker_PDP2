// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "sdram.h"

#include "comms_w.h"
#include "process_w.h"

// this files contains the communication routines used by W cores

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
extern w_conf_t       wcfg;       // weight core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// weight core variables
// ------------------------------------------------------------------------
extern activation_t   * w_outputs[2];  // unit outputs for b-d-p
extern pkt_queue_t      w_delta_pkt_q; // queue to hold received deltas
extern uint             wf_comms;      // pointer to receiving unit outputs
extern scoreboard_t     wf_arrived;    // keeps track of received unit outputs
extern uint             wf_thrds_done; // sync. semaphore: comms, proc & stop
extern uchar            wb_active;     // processing deltas from queue?
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
void w_receivePacket (uint key, uint payload)
{
  // get packet phase
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // check if packet is stop type
  uint stop = ((key & SPINN_STOP_MASK) == SPINN_STPR_KEY);

  // check packet type
  if (stop)
  {
    // stop packet
    w_stopPacket (key, payload);
  }
  else if (ph == SPINN_FORWARD)
  {
    // FORWARD phase
    w_forwardPacket (key, payload);
  }
  else
  {
    // BACKPROP phase
    w_backpropPacket (key, payload);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a stop packet
// ------------------------------------------------------------------------
void w_stopPacket (uint key, uint payload)
{
  #ifdef DEBUG
    stp_recv++;
    if (phase == SPINN_BACKPROP)
      wrng_phs++;
  #endif

  // STOP decision arrived
  tick_stop = (key & SPINN_STPD_MASK) >> SPINN_STPD_SHIFT;

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "sc:%x\n", tick_stop);
  #endif

  // check if all threads done
  if (wf_thrds_done == 0)
  {
    // if done initialize synchronization semaphore,
    wf_thrds_done = 2;

    // and advance tick
    #ifdef TRACE_VRB
      io_printf (IO_BUF, "wrp scheduling wf_advance_tick\n");
    #endif

    spin1_schedule_callback (wf_advance_tick, NULL, NULL, SPINN_WF_TICK_P);
  }
  else
  {
    // if not done report stop thread done
    wf_thrds_done -= 1;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a FORWARD phase packet
// ------------------------------------------------------------------------
void w_forwardPacket (uint key, uint payload)
{
  #ifdef DEBUG
    pkt_recv++;
    recv_fwd++;
    if (phase == SPINN_BACKPROP)
      wrng_phs++;
  #endif

  // get output index: mask out phase, core and block data,
  uint inx = key & SPINN_OUTPUT_MASK;

  // store received unit output,
  w_outputs[wf_comms][inx] = (activation_t) payload;

  // and update scoreboard,
  #if SPINN_USE_COUNTER_SB == FALSE
    wf_arrived |= (1 << inx);
  #else
    wf_arrived++;
  #endif
    
  // if all expected inputs have arrived may move to next tick
  if (wf_arrived == wcfg.f_all_arrived)
  {
    // initialize arrival scoreboard for next tick,
    wf_arrived = 0;

    // update pointer to received unit outputs,
    wf_comms = 1 - wf_comms;

    // and check if other threads are done,
    if (wf_thrds_done == 0)
    {
      // if done initialize synchronization semaphore,
      wf_thrds_done = 2;

      // and advance tick
      #ifdef TRACE_VRB
        io_printf (IO_BUF, "wfpkt scheduling wf_advance_tick\n");
      #endif

      spin1_schedule_callback (wf_advance_tick, NULL, NULL, SPINN_WF_TICK_P);
    }
    else
    {
      // if not done report comms thread done
      wf_thrds_done -= 1;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// enqueue BACKPROP phase packet for later processing
// ------------------------------------------------------------------------
void w_backpropPacket (uint key, uint payload)
{
  #ifdef DEBUG
    pkt_recv++;
    recv_bkp++;
    if (phase == SPINN_FORWARD)
      wrng_phs++;
  #endif

  // check if space in packet queue,
  uint new_tail = (w_delta_pkt_q.tail + 1) % SPINN_WEIGHT_PQ_LEN;

  if (new_tail == w_delta_pkt_q.head)
  {
    // if queue full exit and report failure
    spin1_kill (SPINN_QUEUE_FULL);
  }
  else
  {
    // if not full queue packet,
    w_delta_pkt_q.queue[w_delta_pkt_q.tail].key = key;
    w_delta_pkt_q.queue[w_delta_pkt_q.tail].payload = payload;
    w_delta_pkt_q.tail = new_tail;

    // and schedule processing thread -- if not active already
    //TODO: need to check phase?
    if (!wb_active)
    {
      wb_active = TRUE;
      spin1_schedule_callback (wb_process, NULL, NULL, SPINN_WB_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------
