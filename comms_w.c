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
extern uint bkpKey;               // 32-bit packet ID for backprop passes
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
extern delta_t        * w_deltas[2];   // error deltas for b-d-p
extern uint             wf_comms;      // pointer to receiving unit outputs
extern scoreboard_t     wf_arrived;    // keeps track of received unit outputs
extern uint             wf_thrds_done; // sync. semaphore: comms, proc & stop
extern uint             wb_comms;      // pointer to receiving deltas
extern scoreboard_t     wb_arrived;    // keeps track of received deltas
extern uchar            wb_comms_done; // all expected deltas arrived
extern uchar            wb_procs_done; // current tick error b-d-ps done
//#extern uint             wb_thrds_done; // sync. semaphore: comms, proc & stop
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
// code
// ------------------------------------------------------------------------

// callback routine for a multicast packet received
void w_receivePacket (uint key, uint payload)
{
  // get packet phase
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // check packet type
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
    if (wf_thrds_done == 0)
    {
      // if done initialize synchronization semaphore,
//##      if (tick_stop)
//##        wf_thrds_done = 0;  // no processing and no stop in tick 0 
//##      else
        wf_thrds_done = 2;

      // and advance tick
      #ifdef TRACE
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

// routine to process a forward multicast packet received
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
//##      if (tick_stop)
//##        wf_thrds_done = 0;  // no processing and no stop in tick 0 
//##      else
        wf_thrds_done = 2;

      // and advance tick
      #ifdef TRACE
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

// routine to process a backward multicast packet received
void w_backpropPacket (uint key, uint payload)
{
  #ifdef DEBUG
    pkt_recv++;
    recv_bkp++;
    if (phase == SPINN_FORWARD) wrng_phs++;
    if (wb_comms_done)
    {
      wrng_btk++;
      spin1_kill (SPINN_UNXPD_PKT);
      return;
    }
  #endif

  // get delta index: mask out phase, core and block data,
  uint inx = key & SPINN_DELTA_MASK;

  // store received error delta,
  w_deltas[wb_comms][inx] = (delta_t) payload;

  // and update scoreboard,
  #if SPINN_USE_COUNTER_SB == FALSE
    wb_arrived |= (1 << inx);
  #else
    wb_arrived++;
  #endif

  // if all expected inputs have arrived may move to next tick
  if (wb_arrived == wcfg.b_all_arrived)
  {
    // initialize arrival scoreboard for next tick,
    wb_arrived = 0;

    // update pointer to received deltas,
    wb_comms = 1 - wb_comms;

    // and check synchronization with processing thread
    if (wb_procs_done)
    {
      // if done clear procs synchronization flag,
      wb_procs_done = FALSE;

      // and advance tick
      #ifdef TRACE
        io_printf (IO_BUF, "wbpkt scheduling wb_advance_tick\n");
      #endif

      spin1_schedule_callback (wb_advance_tick, NULL, NULL, SPINN_WB_TICK_P);
    }
    else
    {
      // if not done report comms thread done
      wb_comms_done = TRUE;
    }
  }
}
