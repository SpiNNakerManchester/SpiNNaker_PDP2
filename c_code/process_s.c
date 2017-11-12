// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "mlp_externs.h"

#include "init_s.h"
#include "comms_s.h"
#include "process_s.h"
#include "activation.h"

// set of routines to be used by S core to process data

// ------------------------------------------------------------------------
// process queued packets until queue empty
// ------------------------------------------------------------------------
void s_process (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "s_process\n");
  #endif

  // process packet queue
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

    uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

    // check for an LDS "accumulation" packet
    if ((key & SPINN_TYPE_MASK) == SPINN_LDSA_KEY)
    {
      // process LDS "accumulation" packet
      s_ldsa_packet (key, payload);
    }
    // check for LDS "total" packet
    else if ((key & SPINN_TYPE_MASK) == SPINN_LDST_KEY)
    {
      io_printf (IO_BUF, "Received LDS \"total\" packet: %r\n", payload);
    }
    // else check packet phase and process accordingly
    else if (ph == SPINN_FORWARD)
    {
      #ifdef DEBUG
        recv_fwd++;
        if (phase != SPINN_FORWARD)
          wrng_phs++;
      #endif

      // process FORWARD phase packet
      s_forward_packet (key, payload);
    }
    else
    {
      #ifdef DEBUG
        recv_bkp++;
        if (phase != SPINN_BACKPROP)
          wrng_phs++;
      #endif

      // process BACKPROP phase packet
      s_backprop_packet (key, payload);
    }

    // access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // when done, flag that going to sleep,
  s_active = FALSE;

  // restore interrupts and leave
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process LDSA packet: accumulate the received partial link delta sums
// ------------------------------------------------------------------------
void s_ldsa_packet (uint key, uint payload)
{
  // add the received value to the total so far,
  s_lds_part += (long_lds_t) payload;

  // increment the count of link delta sums arrived,
  s_lds_arrived++;

  // check whether all the partial sums have arrived
  if (s_lds_arrived == scfg.lds_expected)
  {
	// send the result to the first s core
    // to give a total across the whole network
    if (scfg.is_first_group == 0)
    {
      io_printf (IO_BUF, "Epoch %d sending partial link delta sum: ", epoch);
      io_printf (IO_BUF, "%r\n", s_lds_part);
      while (!spin1_send_mc_packet (ldsKey, s_lds_part, WITH_PAYLOAD));
    }

    // access synchronisation semaphore with interrupts disabled
    uint cpsr = spin1_int_disable ();

    // check if all threads done
    if (sb_thrds_done == 0)
    {
      // if done initialise semaphore
      sb_thrds_done = 0;

      // restore interrupts after flag access,
      spin1_mode_restore (cpsr);

      // and advance tick
      //TODO: check if need to schedule or can simply call
      sb_advance_tick (NULL, NULL);
    }
    else
    {
      // if not done report processing thread done,
      sb_thrds_done -= 1;

      // and restore interrupts after flag access
      spin1_mode_restore (cpsr);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process FORWARD phase: accumulate dot products to produce nets
// ------------------------------------------------------------------------
void s_forward_packet (uint key, uint payload)
{
  // get net index: mask out block and phase data,
  uint inx = key & SPINN_NET_MASK;

  // accumulate new net b-d-p,
  // s40.23 = s40.23 + s8.23
  s_nets[inx] += (long_net_t) ((net_t) payload);

  // mark net b-d-p as arrived,
  sf_arrived[inx]++;

  // and check if dot product complete to compute net
  if (sf_arrived[inx] == scfg.fwd_expected)
  {
    net_t net_tmp;

    // saturate and cast the long nets before sending,
    if (s_nets[inx] >= (long_net_t) SPINN_NET_MAX)
    {
      net_tmp = (net_t) SPINN_NET_MAX;
    }
    else if (s_nets[inx] <= (long_net_t) SPINN_NET_MIN)
    {
      net_tmp = (net_t) SPINN_NET_MIN;
    }
    else
    {
      net_tmp = (net_t) s_nets[inx];
    }

    // incorporate net index to the packet key and send,
    while (!spin1_send_mc_packet ((fwdKey | inx), net_tmp, WITH_PAYLOAD));

    #ifdef DEBUG_CFG3
      io_printf (IO_BUF, "sn[%u]: 0x%08x\n", inx, net_tmp);
    #endif

    #ifdef DEBUG
      pkt_sent++;
      sent_fwd++;
    #endif

    // prepare for next tick,
    s_nets[inx] = 0;
    sf_arrived[inx] = 0;

    // mark net as done,
    sf_done++;

    // and check if all nets done
    if (sf_done == scfg.num_units)
    {
      // access synchronization semaphore with interrupts disabled
      uint cpsr = spin1_int_disable ();

      // check if all threads done
      if (sf_thrds_done == 0)
      {
        // if done initialize semaphore
        sf_thrds_done = 1;

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        //TODO: check if need to schedule or can simply call
        sf_advance_tick (NULL, NULL);
      }
      else
      {
        // if not done report processing thread done,
        sf_thrds_done -= 1;

        // and restore interrupts after flag access
        spin1_mode_restore (cpsr);
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP phase: accumulate dot products to produce errors
// ------------------------------------------------------------------------
void s_backprop_packet (uint key, uint payload)
{
  // get error index: mask out block, phase and colour data,
  uint inx = key & SPINN_ERROR_MASK;

  // get error colour: mask out block, phase and net index data,
  uint clr = (key & SPINN_COLOUR_MASK) >> SPINN_COLOUR_SHIFT;

  // accumulate new error b-d-p,
  s_errors[clr][inx] += (error_t) payload;

  // mark error b-d-p as arrived,
  sb_arrived[clr][inx]++;

  // and check if error complete to send to next stage
  if (sb_arrived[clr][inx] == scfg.bkp_expected)
  {
    //NOTE: may need to use long_error_t and saturate before sending
    error_t error = s_errors[clr][inx];

/*
    long_error_t err_tmp = s_errors[clr][inx]
                              >> (SPINN_LONG_ERR_SHIFT - SPINN_ERROR_SHIFT);

    if (err_tmp >= (long_error_t) SPINN_ERROR_MAX)
    {
      error = (error_t) SPINN_ERROR_MAX;
    }
    else if (err_tmp <= (long_error_t) SPINN_ERROR_MIN)
    {
      error = (error_t) SPINN_ERROR_MIN;
    }
    else
    {
      error = (error_t) err_tmp;
    }
*/

    // incorporate error index to the packet key and send,
    while (!spin1_send_mc_packet ((bkpKey | inx), error, WITH_PAYLOAD));

    #ifdef DEBUG_CFG4
      io_printf (IO_BUF, "se[%u]: 0x%08x\n", inx, error);
    #endif

    #ifdef DEBUG
      pkt_sent++;
      sent_bkp++;
    #endif

    // prepare for next tick,
    s_errors[clr][inx] = 0;
    sb_arrived[clr][inx] = 0;

    // mark error as done,
    sb_done++;

    // and check if all errors done
    if (sb_done == scfg.num_units)
    {
      // access synchronization semaphore with interrupts disabled
      uint cpsr = spin1_int_disable ();

      // check if all threads done
      if (sb_thrds_done == 0)
      {
        // if done initialize semaphore:
        // if we are using Doug's Momentum, and we have reached the end of the
        // epoch (i.e. we are on the last example, and are about to move on to
        // the last tick, we need have to wait for the partial link delta sums
        // to arrive, so increment the thread count by 1
        if (scfg.update_function == SPINN_DOUGSMOMENTUM_UPDATE
            && example == (ncfg.num_examples - 1)
            && tick == SPINN_SB_END_TICK + 1)
        {
          sb_thrds_done = 1;
        }
        else
        {
          sb_thrds_done = 0;
        }
        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        //TODO: check if need to schedule or can simply call
        sb_advance_tick (NULL, NULL);
      }
      else
      {
        // if not done report processing thread done,
        sb_thrds_done -= 1;

        // and restore interrupts after flag access
        spin1_mode_restore (cpsr);
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void sf_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "sf_advance_tick\n");
  #endif

  // prepare for next tick,
  sf_done = 0;

  #ifdef DEBUG
    tot_tick++;
  #endif

  // and check if end of example's FORWARD phase
  if (tick_stop)
  {
    sf_advance_event ();
  }
  else
  {
    // if not done increment tick
    tick++;

    #ifdef DEBUG
      io_printf (IO_BUF, "sf_tick: %d/%d\n", tick, tot_tick);
    #endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void sb_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "sb_advance_tick\n");
  #endif

  // prepare for next tick,
  sb_done = 0;

  #ifdef DEBUG
    tot_tick++;
  #endif

  // and check if end of BACKPROP phase
  if (tick == SPINN_SB_END_TICK)
  {
    // initialize the tick count
    tick = SPINN_S_INIT_TICK;

    #ifdef TRACE
      io_printf (IO_BUF, "s_switch_to_fw\n");
    #endif

    // switch to FORWARD phase,
    phase = SPINN_FORWARD;

    // and move to next example
    s_advance_example ();
  }
  else
  {
    // if not done decrement tick
    tick--;

    #ifdef DEBUG
      io_printf (IO_BUF, "sb_tick: %d/%d\n", tick, tot_tick);
    #endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: update the event at the end of a simulation tick
// ------------------------------------------------------------------------
void sf_advance_event (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "sf_advance_event\n");
  #endif

  // check if done with ticks
  if (tick == ncfg.global_max_ticks - 1)
  {
    evt = num_events - 1;
  }

  // check if done with events
  if (++evt >= num_events)
  {
    // and check if in training mode
    if (ncfg.training)
    {
      // if training save number of ticks,
      num_ticks = tick;

      // and do BACKPROP phase
      phase = SPINN_BACKPROP;
    }
    else
    {
      // if not training initialize ticks,
      tick = SPINN_S_INIT_TICK;

      // and move to next example
      s_advance_example ();
    }
  }
  else
  {
    // if not done increment tick
    tick++;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: update the example at the end of a simulation tick
// ------------------------------------------------------------------------
void s_advance_example (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "s_advance_example\n");
  #endif

  // check if done with examples
  if (++example >= ncfg.num_examples)
  {
    // check if done with epochs
    if (++epoch >= ncfg.num_epochs)
    {
      // done
      spin1_exit (SPINN_NO_ERROR);
      return;
    }
    else
    {
      // start from first example again
      example = 0;

      // reset the partial link delta sum
      if (ncfg.training)
      {
        s_lds_part = 0;
        s_lds_arrived = 0;
      }
    }
  }

  // start from first event for next example
  evt = 0;
  num_events = ex[example].num_events;
}
// ------------------------------------------------------------------------
