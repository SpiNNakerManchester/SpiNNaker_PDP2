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

    // and check packet phase and process accordingly
    uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;
    if (ph == SPINN_FORWARD)
    {
      // process FORWARD phase packet
      #ifdef DEBUG
        recv_fwd++;
        if (phase != SPINN_FORWARD)
          wrng_phs++;
      #endif

      s_forward_packet (key, payload);
    }
    else
    {
      // process BACKPROP phase packet
      #ifdef DEBUG
        recv_bkp++;
        if (phase != SPINN_BACKPROP)
          wrng_phs++;
      #endif

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
// process FORWARD phase: accumulate dot products to produce nets
// ------------------------------------------------------------------------
void s_forward_packet (uint key, uint payload)
{
  // get net index: mask out block, phase and colour data,
  uint inx = key & SPINN_NET_MASK;

  // get net colour: mask out block, phase and net index data,
  uint clr = ((key & SPINN_COLOUR_MASK) >> SPINN_COLOUR_SHIFT);

  // accumulate new net b-d-p,
  // s40.23 = s40.23 + s8.23
  s_nets[clr][inx] += (long_net_t) ((net_t) payload);

  // mark net b-d-p as arrived,
  #if SPINN_USE_COUNTER_SB == FALSE
    // get net block: mask out phase, colour and net index data
    uint blk = (key & SPINN_BLK_R_MASK) >> SPINN_BLK_R_SHIFT;

    // check if already marked -- problem,
    #ifdef DEBUG
      if (sf_arrived[clr][inx] & (1 << blk))
      {
        io_printf (IO_BUF, "!c:%u b:%u k:%u a:0x%08x\n",
                    clr, blk, inx, sf_arrived[clr][inx]
                  );
      }
    #endif

    // mark it
    sf_arrived[clr][inx] |= (1 << blk);
  #else
    sf_arrived[clr][inx]++;
  #endif

  // and check if dot product complete to compute net
  if (sf_arrived[clr][inx] == scfg.f_all_arrived)
  {
    net_t net_tmp;

    // saturate and cast the long nets before sending,
    if (s_nets[clr][inx] >= (long_net_t) SPINN_NET_MAX)
    {
      net_tmp = (net_t) SPINN_NET_MAX;
    }
    else if (s_nets[clr][inx] <= (long_net_t) SPINN_NET_MIN)
    {
      net_tmp = (net_t) SPINN_NET_MIN;
    }
    else
    {
      net_tmp = (net_t) s_nets[clr][inx];
    }

    // incorporate net index to the packet key and send,
    while (!spin1_send_mc_packet ((fwdKey | inx), net_tmp, WITH_PAYLOAD));

    #ifdef DEBUG
      pkt_sent++;
      sent_fwd++;
    #endif

    // prepare for next tick,
    s_nets[clr][inx] = 0;
    sf_arrived[clr][inx] = 0;

    // mark net as done,
    #if SPINN_USE_COUNTER_SB == FALSE
      sf_done |= (1 << inx);
    #else
      sf_done++;
    #endif

    // and check if all nets done
    if (sf_done == scfg.f_all_done)
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
  #if SPINN_USE_COUNTER_SB == FALSE
    // get error block: mask out phase, colour and net index data
    uint blk = (key & SPINN_BLK_C_MASK) >> SPINN_BLK_C_SHIFT;

    // check if already marked -- problem,
    #ifdef DEBUG
      if (sb_arrived[clr][inx] & (1 << blk))
      {
        io_printf (IO_BUF, "!c:%u b:%u k:%u a:0x%08x\n",
		        clr, blk, inx, sb_arrived[clr][inx]
                  );
      }
    #endif

      // mark it
    sb_arrived[clr][inx] |= (1 << blk);
  #else
    sb_arrived[clr][inx]++;
  #endif

  // and check if error complete to send to next stage
  //TODO: can use a configuration constant -- needs fixing
  if (sb_arrived[clr][inx] == sb_all_arrived)
  {
    //NOTE: may need to use long_error_t and saturate before sending
    error_t error = s_errors[clr][inx];

/*    long_error_t err_tmp = s_errors[clr][inx]
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
    }*/

    // incorporate error index to the packet key and send,
    while (!spin1_send_mc_packet ((bkpKey | inx), error, WITH_PAYLOAD));

    #ifdef DEBUG
      pkt_sent++;
      sent_bkp++;
    #endif

    // prepare for next tick,
    s_errors[clr][inx] = 0;
    sb_arrived[clr][inx] = 0;

    // mark error as done,
    #if SPINN_USE_COUNTER_SB == FALSE
      sb_done |= (1 << inx);
    #else
      sb_done++;
    #endif

    // and check if all errors done
    if (sb_done == scfg.b_all_done)
    {
      // advance tick
      //TODO: check if need to schedule or can simply call
      sb_advance_tick (NULL, NULL);
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
      io_printf (IO_BUF, "w_switch_to_fw\n");
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

  // check if done with events
  if (++evt >= num_events)
  {
    // and check if in training mode
    if (ncfg.training)
    {
      // if training save number of ticks,
      num_ticks = tick;

      #ifdef TRACE
        io_printf (IO_BUF, "w_switch_to_bp\n");
      #endif

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
    }
  }

  // start from first event for next example
  evt = 0;
  num_events = ex[example].num_events;
}
// ------------------------------------------------------------------------
