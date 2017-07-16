// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_w.h"
#include "process_w.h"

// this files contains the communication routines used by W cores

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

  // store output for use in backprop phase,
  if (tick > 0)
  {
    store_outputs (inx);
  }

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
    spin1_exit (SPINN_QUEUE_FULL);
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
// ------------------------------------------------------------------------+


// ------------------------------------------------------------------------
// stores the outputs received for the current tick
// ------------------------------------------------------------------------
void store_outputs (uint inx)
{
  #ifdef TRACE
    io_printf (IO_BUF, "store_outputs\n");
  #endif

  w_output_history[(tick * wcfg.num_rows) + inx] = w_outputs[wf_comms][inx];
}
// ------------------------------------------------------------------------
