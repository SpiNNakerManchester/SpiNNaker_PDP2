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


// ------------------------------------------------------------------------
// sum core computation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process FORWARD phase: accumulate dot products to produce nets
// ------------------------------------------------------------------------
void sf_process (uint key, uint payload)
{
#ifdef DEBUG
  recv_fwd++;
  if (phase != SPINN_FORWARD)
    wrng_phs++;
#endif

  // get net index: mask out block and phase data,
  uint inx = key & SPINN_NET_MASK;

  // get error colour: mask out block, phase and net index data,
  uint clr = (key & SPINN_COLOUR_MASK) >> SPINN_COLOUR_SHIFT;

  // accumulate new net b-d-p,
  s_nets[clr][inx] += (long_net_t) ((net_t) payload);

  // mark net b-d-p as arrived,
  sf_arrived[clr][inx]++;

  // and check if dot product complete to compute net
  if (sf_arrived[clr][inx] == scfg.fwd_expected)
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
    sf_done++;

    // and check if all nets done
    if (sf_done == scfg.num_units)
    {
       // prepare for next tick,
       sf_done = 0;

      // access thread semaphore with interrupts disabled
      uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
      if (!(sf_thrds_pend & SPINN_THRD_PROC))
        wrng_pth++;
#endif

      // and check if all other threads done
      if (sf_thrds_pend == SPINN_THRD_PROC)
      {
        // if done initialise semaphore
        sf_thrds_pend = SPINN_SF_THRDS;

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        sf_advance_tick ();
      }
      else
      {
        // if not done report processing thread done,
        sf_thrds_pend &= ~SPINN_THRD_PROC;

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
void sb_process (uint key, uint payload)
{
#ifdef DEBUG
  recv_bkp++;
  if (phase != SPINN_BACKPROP)
    wrng_phs++;
#endif

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
      // prepare for next tick,
      sb_done = 0;

      // access thread semaphore with interrupts disabled
      uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
      if (!(sb_thrds_pend & SPINN_THRD_PROC))
        wrng_pth++;
#endif

      // check if all other threads done
      if (sb_thrds_pend == SPINN_THRD_PROC)
      {
        // if done initialise semaphore:
        // if we are using Doug's Momentum, and we have reached the end of the
        // epoch (i.e. we are on the last example, and are about to move on to
        // the last tick, we need have to wait for the partial link delta sums
        // to arrive
        //TODO: find a better place to do this calculation
        if (xcfg.update_function == SPINN_DOUGSMOMENTUM_UPDATE
            && example_cnt == (xcfg.num_examples - 1)
            && tick == SPINN_SB_END_TICK + 1)
        {
          // if this s core relates to the first group in the network, then we
          // also need to wait for the link delta sum totals
          if (scfg.is_first_group)
          {
            sb_thrds_pend = SPINN_SB_THRDS | SPINN_THRD_LDSA | SPINN_THRD_LDST;
          }
          else
          {
            sb_thrds_pend = SPINN_SB_THRDS | SPINN_THRD_LDSA;
          }
        }

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        sb_advance_tick ();
      }
      else
      {
        // if not done report processing thread done,
        sb_thrds_pend &= ~SPINN_THRD_PROC;

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
void sf_advance_tick (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "sf_advance_tick\n");
#endif

#ifdef DEBUG
  tot_tick++;
#endif

  // check if end of event
  if (tick_stop)
  {
    sf_advance_event ();
  }
  else
  {
    // if not done increment tick
    tick++;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void sb_advance_tick (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "sb_advance_tick\n");
#endif

#ifdef DEBUG
  tot_tick++;
#endif

  // check if end of BACKPROP phase
  if (tick == SPINN_SB_END_TICK)
  {
    // initialise the tick count
    tick = SPINN_S_INIT_TICK;

    // switch to FORWARD phase,
    phase = SPINN_FORWARD;

    // and move to next example
    s_advance_example ();
  }
  else
  {
    // if not done decrement tick
    tick--;
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

  // check if done with example's FORWARD phase
  if ((++evt >= num_events) || (tick == ncfg.global_max_ticks - 1))
  {
    // check if in training mode
    if (xcfg.training)
    {
      // move on to BACKPROP phase
      phase = SPINN_BACKPROP;
    }
    else
    {
      // if not training initialise ticks,
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
// update example at the end of a (FORWARD or BACKPROP) tick
// ------------------------------------------------------------------------
void s_advance_example (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "s_advance_example\n");
#endif

  // point to next example in the set - wrap around if at the end,
  if (++example_inx >= es->num_examples)
  {
    example_inx = 0;
  }

  // check if done with examples,
  if (++example_cnt >= xcfg.num_examples)
  {
    // prepare for next epoch,
    epoch++;

    // access network stop flag with interrupts disabled,
    uint cpsr = spin1_int_disable ();

    // check if network stop decision ready,
    if (net_stop_rdy)
    {
      // clear flag,
      net_stop_rdy = FALSE;

      // restore interrupts,
      spin1_mode_restore (cpsr);

      // and decide what to do
      if (net_stop)
      {
        // and finish stage - report no error
        //TODO: check if need to schedule or can simply call
        spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
      }
    }
    else
    {
      // flag ready for net_stop decision,
      net_stop_rdy = TRUE;

      // and restore interrupts
      spin1_mode_restore (cpsr);
    }

    // reset example count for next epoch,
    example_cnt = 0;

    // and reset the partial link delta sum
    if (xcfg.training)
    {
      s_lds_part = 0;
      s_ldsa_arrived = 0;
      s_ldst_arrived = 0;
    }
  }

  // start from first event for next example,
  evt = 0;
  num_events = ex[example_inx].num_events;

  // and send sync packet to allow next example to start
  while (!spin1_send_mc_packet (fdsKey, 0, NO_PAYLOAD));

#ifdef DEBUG
  pkt_sent++;
  spk_sent++;
#endif
}
// ------------------------------------------------------------------------
