// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"


// ------------------------------------------------------------------------
// threshold core communications routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process received packets (stop, chain, sync, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void t_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

  // check if packet is stop type
  uint stop = ((key & SPINN_TYPE_MASK) == SPINN_STOP_KEY);
  if (stop)
  {
    // process stop final decision packet
    t_stopPacket (key);
    return;
  }

  // check if packet is chain type
  uint chain = ((key & SPINN_TYPE_MASK) == SPINN_STPC_KEY);
  if (chain)
  {
    // process stop decision chain packet
    t_chainPacket (key);
    return;
  }

  // check if packet is network stop type
  uint stpn = ((key & SPINN_TYPE_MASK) == SPINN_STPN_KEY);
  if (stpn)
  {
    // process network stop decision packet
    t_networkStopPacket ();
    return;
  }

  // get packet phase
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // check if packet is sync type
  uint sync = ((key & SPINN_TYPE_MASK) == SPINN_SYNC_KEY);
  if (sync)
  {
    // process tick synchronisation packet
    t_syncPacket (ph);
    return;
  }

  // computation packet
    if (ph == SPINN_FORWARD)
  {
    // FORWARD phase packet
    t_forwardPacket (key, payload);
  }
  else
  {
    // BACKPROP phase packet
    t_backpropPacket (key, payload);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a stop final decision packet
// ------------------------------------------------------------------------
void t_stopPacket (uint key)
{
#ifdef DEBUG
  stp_recv++;
#endif

  // STOP decision arrived
  tick_stop = key & SPINN_STPD_MASK;

#ifdef DEBUG_VRB
  io_printf (IO_BUF, "sc:%x\n", tick_stop);
#endif

  // check if all other threads done
  if (tf_thrds_pend == 0)
  {
    // initialise semaphore
    tf_thrds_pend = 1;

    // and advance tick
    spin1_schedule_callback (tf_advance_tick, 0, 0, SPINN_TF_TICK_P);
  }
  else
  {
    // if not done report stop thread done
    tf_thrds_pend -= 1;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a stop daisy chain packet
// ------------------------------------------------------------------------
void t_chainPacket (uint key)
{
#ifdef DEBUG
  chn_recv++;
#endif

  // STOP daisy chain partial decision arrived from previous core
  tf_chain_prev = key & SPINN_STPD_MASK;

  // check if chain value can be forwarded
  if (tf_chain_rdy)
  {
    // initialise flag,
    tf_chain_rdy = tf_initChain;

    // report outputs to host if requested,
    if (tcfg.write_out)
    {
      spin1_schedule_callback (send_outputs_to_host, SPINN_HOST_NORMAL, tick,
			       SPINN_T_SEND_OUTS_P);
    }

    // send stop packet,
    spin1_schedule_callback (tf_send_stop, 0, 0, SPINN_T_SEND_STOP_P);

    // and advance tick if last_output_group
    //NOTE: this group does not get a stop decision packet
    //      so it's ready to advance tick
    if (tcfg.is_last_output_group)
    {
      spin1_schedule_callback (tf_advance_tick, 0, 0, SPINN_TF_TICK_P);
    }
  }
  else
  {
    // if not, flag that the previous value arrived
    tf_chain_rdy = 1;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop decision packet
// ------------------------------------------------------------------------
void t_networkStopPacket (void)
{
#ifdef DEBUG
  stn_recv++;
#endif

  // report no error
  stage_done (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// process a sync packet
// ------------------------------------------------------------------------
void t_syncPacket (uint ph)
{
#ifdef DEBUG
  spk_recv++;
#endif

  if (ph == SPINN_FORWARD)
  {
    // keep track of FORWARD sync packets,
    t_sync_arrived++;

    // and check if all expected packets arrived
    if (t_sync_arrived == tcfg.fwd_sync_expected)
    {
      // initialise for next synchronisation,
      t_sync_arrived = 0;

      // and check if can trigger sending data
      if (t_sync_rdy)
      {
        // clear synchronisation flag,
        t_sync_rdy = FALSE;

        // schedule sending of unit outputs to w cores,
        spin1_schedule_callback (t_init_outputs,
                                  0, 0, SPINN_T_INIT_OUT_P
                                );

        // and, if required, send outputs to host
        if (tcfg.write_out)
        {
          spin1_schedule_callback (send_outputs_to_host,
                                    SPINN_HOST_NORMAL, 0, SPINN_T_SEND_OUTS_P
                                  );
        }
      }
      else
      {
        // if not flag sync as ready
        t_sync_rdy = TRUE;
      }
    }
  }
/*
  //NOTE: no longer using BACKPROP synchronisation packets
  else
  {
    // keep track of BACKPROP sync packets,
    t_sync_arrived++;

    // and check if all expected packets arrived,
    if (t_sync_arrived == tcfg.bkp_sync_expected)
    {
      // initialise for next synchronisation,
      t_sync_arrived = 0;

      // check if can trigger sending data
      if (phase == SPINN_BACKPROP)
      {
        // schedule sending of deltas
        //#spin1_schedule_callback (t_init_deltas, 0, 0, SPINN_T_INIT_DLT_P);
      }
      else
      {
        // if not ready flag sync done
        t_sync_rdy = TRUE;
      }
    }
  }
*/
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// enqueue FORWARD phase packet for later processing
// ------------------------------------------------------------------------
void t_forwardPacket (uint key, uint payload)
{
#ifdef DEBUG
  recv_fwd++;
  if (phase == SPINN_BACKPROP)
    wrng_phs++;
#endif

  // check if space in FORWARD packet queue,
  uint new_tail = (t_net_pkt_q.tail + 1) % SPINN_THLD_PQ_LEN;

  if (new_tail == t_net_pkt_q.head)
  {
    // report queue full error
    stage_done (SPINN_QUEUE_FULL);
  }
  else
  {
    // if not full queue packet,
    t_net_pkt_q.queue[t_net_pkt_q.tail].key = key;
    t_net_pkt_q.queue[t_net_pkt_q.tail].payload = payload;
    t_net_pkt_q.tail = new_tail;

    // and schedule processing thread
    // if in FORWARD phase and not active already
    //TODO: need to check phase?
    if ((phase == SPINN_FORWARD) && (!t_active))
    {
      t_active = TRUE;
      spin1_schedule_callback (tf_process, 0, 0, SPINN_TF_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a BACKPROP phase packet
// ------------------------------------------------------------------------
void t_backpropPacket (uint key, uint payload)
{
#ifdef DEBUG
  recv_bkp++;
  if (phase == SPINN_FORWARD)
    wrng_phs++;
#endif

  // get error index: mask out phase, core and block data,
  uint inx = key & SPINN_ERROR_MASK;

  // store received error,
  t_errors[tb_comms][inx] = (error_t) payload;

  // and update scoreboard,
  tb_arrived++;

  // if all expected errors have arrived may move to next tick
  if (tb_arrived == tcfg.num_units)
  {
    // initialise arrival scoreboard for next tick,
    tb_arrived = 0;

    // update pointer to received errors,
    tb_comms = 1 - tb_comms;

    // and check if all other threads are done,
    if (tb_thrds_pend == 0)
    {
      // if done initialise synchronisation semaphore,
      tb_thrds_pend = 1;

      // and advance tick
#ifdef TRACE_VRB
      io_printf (IO_BUF, "tbpkt scheduling tb_advance_tick\n");
#endif

      spin1_schedule_callback (tb_advance_tick, 0, 0, SPINN_TB_TICK_P);
    }
    else
    {
      // if not done report comms thread done
      tb_thrds_pend -= 1;
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// send relevant data to host using SDP messages
// TODO: all outputs may not fit in one SDP message!
// ------------------------------------------------------------------------
void send_outputs_to_host (uint cmd, uint tick)
{
  // adjust event according to Lens reporting,
  int le = (tick == 0) ? -1 : (int) evt;

  // report epoch, example and tick,
  t_sdp_msg.cmd_rc = cmd;
  t_sdp_msg.seq    = tcfg.write_blk;
  t_sdp_msg.arg1   = epoch;
  t_sdp_msg.arg2   = (le << 16) | example;
  t_sdp_msg.arg3   = tick;

  // set default message data length (no data)
  uint len = 0;

  if (cmd == SPINN_HOST_NORMAL)
  {
    // copy outputs and targets into msg buffer,
    short_activ_t * my_data = (short_activ_t *) t_sdp_msg.data;
    for (uint i = 0; i < tcfg.num_units; i++)
    {
      if (tick == 0)
      {
        my_data[2 * i]     = 0;
        my_data[2 * i + 1] = 0;
      }
      else
      {
        my_data[2 * i]     = (short_activ_t) (t_outputs[i]
                >> (SPINN_ACTIV_SHIFT
              - SPINN_SHORT_ACTIV_SHIFT));
        if (tt[t_it_idx + i] == SPINN_ACTIV_ONE)
        {
          my_data[2 * i + 1] = SPINN_SHORT_ACTIV_MAX;
        }
        else
        {
          my_data[2 * i + 1] = (short_activ_t) (tt[t_it_idx + i]
                  >> (SPINN_ACTIV_SHIFT
                - SPINN_SHORT_ACTIV_SHIFT));
        }
      }
    }

    // and set message data length,
    len = 2 * tcfg.num_units * sizeof (short_activ_t);
  }

  // set message length,
  t_sdp_msg.length = sizeof (sdp_hdr_t) + sizeof (cmd_hdr_t) + len;

  // and send message
  while (!spin1_send_sdp_msg (&t_sdp_msg, SPINN_SDP_TMOUT));
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// send an sdp packet to the host with information related to
// various parameters of the simulation: id of the output group sending the
// data, number of output units, number of groups writing outputs and number
// of ticks of simulation
// ------------------------------------------------------------------------
void send_info_to_host (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

  // send initial info to host
  // report epoch, example and tick,
  t_sdp_msg.cmd_rc = SPINN_HOST_INFO;
  t_sdp_msg.seq    = tcfg.write_blk;
  t_sdp_msg.arg1   = tcfg.num_units;
  t_sdp_msg.arg2   = ncfg.num_write_blks;
  t_sdp_msg.arg3   = t_tot_ticks + 1;

  // copy initial outputs and targets into msg buffer,
  short_activ_t * my_data = (short_activ_t *) t_sdp_msg.data;
  for (uint i = 0; i < tcfg.num_units; i++)
  {
    my_data[2 * i]     = 0;
    my_data[2 * i + 1] = 0;
  }

  // set message length,
  uint len = 2 * tcfg.num_units * sizeof (short_activ_t);
  t_sdp_msg.length = sizeof (sdp_hdr_t) + sizeof (cmd_hdr_t) + len;

  // and send message
  while (!spin1_send_sdp_msg (&t_sdp_msg, SPINN_SDP_TMOUT));

#ifdef DEBUG_VRB
  io_printf (IO_BUF, "sent info to host: nb:%d wb:%d no:%d tt:%d\n",
	     ncfg.num_write_blks, tcfg.write_blk,
	     tcfg.num_units, t_tot_ticks
    );
#endif
}
// ------------------------------------------------------------------------
