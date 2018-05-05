// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_w.h"
#include "process_w.h"

// this file contains the communication routines used by W cores

// ------------------------------------------------------------------------
// process received packets (stop, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void w_receivePacket (uint key, uint payload)
{
  // check if packet is stop type
  uint stop = ((key & SPINN_TYPE_MASK) == SPINN_STOP_KEY);
  if (stop)
  {
    // process stop packet
    w_stopPacket (key);
    return;
  }

  // check if packet is network stop type
  uint stpn = ((key & SPINN_TYPE_MASK) == SPINN_STPN_KEY);
  if (stpn)
  {
    // process network stop decision packet
    w_networkStopPacket ();
    return;
  }

  // check if packet is ldsr type
  uint ldsr = ((key & SPINN_TYPE_MASK) == SPINN_LDSR_KEY);
  if (ldsr)
  {
    // process ldsr packet
    w_ldsrPacket (payload);
    return;
  }

  // computation packet - get packet phase and block
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;
  uint blk = (key & SPINN_BLOCK_MASK) >> SPINN_BLOCK_SHIFT;
  if (ph == SPINN_FORWARD)
  {
    // FORWARD phase packet in my block
	if (blk == wcfg.row_blk)
	{
	  w_forwardPacket (key, payload);
	}
  }
  else
  {
    // BACKPROP phase packet in my block
	if (blk == wcfg.col_blk)
	{
	  w_backpropPacket (key, payload);
	}
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a stop packet
// ------------------------------------------------------------------------
void w_stopPacket (uint key)
{
  #ifdef DEBUG
    stp_recv++;
    if (phase == SPINN_BACKPROP)
      wrng_phs++;
  #endif

  // STOP decision arrived
  tick_stop = key & SPINN_STPD_MASK;

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "sc:%x\n", tick_stop);
  #endif

  // check if all other threads done
  if (wf_thrds_pend == 0)
  {
    // if done initialize synchronization semaphore,
    wf_thrds_pend = 2;

    // and advance tick
    #ifdef TRACE_VRB
      io_printf (IO_BUF, "wrp scheduling wf_advance_tick\n");
    #endif

    spin1_schedule_callback (wf_advance_tick, NULL, NULL, SPINN_WF_TICK_P);
  }
  else
  {
    // if not done report stop thread done
    wf_thrds_pend -= 1;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop decision packet
// ------------------------------------------------------------------------
void w_networkStopPacket (void)
{
  #ifdef DEBUG
    stn_recv++;
  #endif

  //done
  spin1_exit (SPINN_NO_ERROR);
  return;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process an ldsr packet
// ------------------------------------------------------------------------
void w_ldsrPacket (uint payload)
{
  // the final link delta sum for the epoch arrived
  w_lds_final = (lds_t) payload;

  // check if all other threads done
  if (wb_thrds_pend == 0)
  {
    //NOTE: no need to initialize semaphore
    //wb_thrds_pend = 0;

    #ifdef TRACE_VRB
      io_printf (IO_BUF, "ldsr calling wb_advance_tick\n");
    #endif

    // and advance tick
    //TODO: check if need to schedule or can simply call
    spin1_schedule_callback (wb_advance_tick, NULL, NULL, SPINN_WB_TICK_P);
  }
  else
  {
    // if not done report processing thread done,
    wb_thrds_pend -= 1;
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
  uint inx = key & SPINN_BLKOUT_MASK;

  // store received unit output,
  w_outputs[wf_comms][inx] = (activation_t) payload;

  // store output for use in backprop phase,
  if (tick > 0)
  {
    store_output (inx);
  }

  // and update scoreboard,
  wf_arrived++;

  // if all expected inputs have arrived may move to next tick
  if (wf_arrived == wcfg.num_rows)
  {
    // initialize arrival scoreboard for next tick,
    wf_arrived = 0;

    // update pointer to received unit outputs,
    wf_comms = 1 - wf_comms;

    // and check if all other threads are done,
    if (wf_thrds_pend == 0)
    {
      // if done initialize synchronization semaphore,
      wf_thrds_pend = 2;

      // and advance tick
      #ifdef TRACE_VRB
        io_printf (IO_BUF, "wfpkt scheduling wf_advance_tick\n");
      #endif

      spin1_schedule_callback (wf_advance_tick, NULL, NULL, SPINN_WF_TICK_P);
    }
    else
    {
      // if not done report comms thread done
      wf_thrds_pend -= 1;
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
// stores output received for the current tick
// ------------------------------------------------------------------------
void store_output (uint inx)
{
  #ifdef TRACE
    io_printf (IO_BUF, "store_output\n");
  #endif

  w_output_history[(tick * wcfg.num_rows) + inx] = w_outputs[wf_comms][inx];
}
// ------------------------------------------------------------------------
