// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_t.h"
#include "process_t.h"

// this file contains the communication routines used by T cores

// ------------------------------------------------------------------------
// process received packets (stop, chain, sync, FORWARD and BACKPROP types)
// ------------------------------------------------------------------------
void t_receivePacket (uint key, uint payload)
{
  // get packet phase
  uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

  // packet is stop type
  uint stop = ((key & SPINN_TYPE_MASK) == SPINN_STOP_KEY);

  // packet is chain type
  uint chain = ((key & SPINN_TYPE_MASK) == SPINN_STPC_KEY);

  // packet is network stop type
  uint stpn = ((key & SPINN_TYPE_MASK) == SPINN_STPN_KEY);

  // packet is sync type
  uint sync = ((key & SPINN_TYPE_MASK) == SPINN_SYNC_KEY);

  // check packet type
  if (stop)
  {
    // stop final decision packet
    t_stopPacket (key);
  }
  else if (chain)
  {
    // stop decision chain packet
    t_chainPacket (key);
  }
  else if (stpn)
  {
    // network stop decision packet
    t_networkStopPacket (key);
  }
  else if (sync)
  {
    // tick synchronisation packet
    t_syncPacket (ph);
  }
  else if (ph == SPINN_FORWARD)
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

  // check if all threads done
  if (tf_thrds_done == 0)
  {
    // initialise semaphore
    tf_thrds_done = 1;

    // and advance tick
    spin1_schedule_callback (tf_advance_tick, NULL, NULL, SPINN_TF_TICK_P);
  }
  else
  {
    // if not done report stop thread done
    tf_thrds_done -= 1;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a stop daisy chain packet
// ------------------------------------------------------------------------
void t_chainPacket (uint key)
{
  #ifdef DEBUG
    stp_recv++;
  #endif

  // STOP daisy chain partial decision arrived from previous core
  tf_chain_prev = key & SPINN_STPD_MASK;

  // check if chain value can be forwarded
  if (tf_chain_rdy)
  {
    // initialise flag,
    tf_chain_rdy = tf_chain_init;

    // and send stop packet
    spin1_schedule_callback (tf_send_stop, NULL, NULL, SPINN_SEND_STOP_P);

    // last group in the chain does not get a stop decision packet
    // so it's ready to advance tick
    if (tcfg.is_last_output_group)
    {
      // advance tick
      spin1_schedule_callback (tf_advance_tick, NULL, NULL, SPINN_TF_TICK_P);
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
void t_networkStopPacket (uint key)
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
      if (phase == SPINN_FORWARD)
      {
        // schedule sending of unit outputs
        spin1_schedule_callback (t_init_outputs,
                                  NULL, NULL, SPINN_T_INIT_OUT_P
                                );

        // and, if required, send outputs to host
        if (tcfg.write_out)
        {
          spin1_schedule_callback (send_outputs_to_host,
                                    SPINN_HOST_NORMAL, 0, SPINN_SEND_OUTS_P
                                  );
        }
      }
      else
      {
        // if not ready flag sync done
        t_sync_done = TRUE;
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
        //#spin1_schedule_callback (t_init_deltas, NULL, NULL, SPINN_SEND_DELTAS_P);
      }
      else
      {
        // if not ready flag sync done
        t_sync_done = TRUE;
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
    pkt_recv++;
    recv_fwd++;
    if (phase == SPINN_BACKPROP)
      wrng_phs++;
  #endif

  // check if space in FORWARD packet queue,
  uint new_tail = (t_net_pkt_q.tail + 1) % SPINN_THLD_PQ_LEN;

  if (new_tail == t_net_pkt_q.head)
  {
    // if queue full exit and report failure
    spin1_exit (SPINN_QUEUE_FULL);
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
      spin1_schedule_callback (tf_process, NULL, NULL, SPINN_TF_PROCESS_P);
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
    pkt_recv++;
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

    // and check if other threads are done,
    if (tb_thrds_done == 0)
    {
      // if done initialise synchronisation semaphore,
      tb_thrds_done = 1;

      // and advance tick
      #ifdef TRACE_VRB
        io_printf (IO_BUF, "tbpkt scheduling tb_advance_tick\n");
      #endif

      spin1_schedule_callback (tb_advance_tick, NULL, NULL, SPINN_TB_TICK_P);
    }
    else
    {
      // if not done report comms thread done
      tb_thrds_done -= 1;
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
  int le;
  le = (tick == 0) ? -1 : (int) evt;

  // report epoch, example and tick,
  t_sdp_msg.cmd_rc = cmd;
  t_sdp_msg.seq    = tcfg.write_blk;
  t_sdp_msg.arg1   = epoch;
  t_sdp_msg.arg2   = (le << 16) | example;
  t_sdp_msg.arg3   = tick;

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
      my_data[2 * i]     = (short_activ_t) (t_outputs[i] >> (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT));
      if (tt[t_it_idx + i] == SPINN_ACTIV_ONE)
      {
        my_data[2 * i + 1] = SPINN_SHORT_ACTIV_MAX;
      }
      else
      {
        my_data[2 * i + 1] = (short_activ_t) (tt[t_it_idx + i] >> (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT));
      }
    }
  }

  // set message length,
  uint len = 2 * tcfg.num_units * sizeof(short_activ_t);
  t_sdp_msg.length = sizeof (sdp_hdr_t) + sizeof (cmd_hdr_t) + len;

  // and send message
  while (!spin1_send_sdp_msg (&t_sdp_msg, SPINN_SDP_TMOUT))
    io_printf (IO_STD, "sdp!\n");
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// send an sdp packet to the host with information related to
// various parameters of the simulation: id of the output group sending the
// data, number of output units, number of units writing outputs an dnumber of
// ticks of simulation
// ------------------------------------------------------------------------
void send_info_to_host (uint null0, uint null1)
{
  // send initial info to host
  // report epoch, example and tick,
  t_sdp_msg.cmd_rc = SPINN_HOST_INFO;
  t_sdp_msg.seq    = tcfg.write_blk;
  t_sdp_msg.arg1   = tcfg.num_units;
  t_sdp_msg.arg2   = ncfg.num_write_blks;
  t_sdp_msg.arg3   = t_tot_ticks + 1;

  // set message length,
  t_sdp_msg.length = sizeof (sdp_hdr_t) + sizeof (cmd_hdr_t);

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
