// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "mlp_externs.h"

#include "init_i.h"
#include "comms_i.h"
#include "process_i.h"
#include "activation.h"


// ------------------------------------------------------------------------
// input core computation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process FORWARD phase: apply input pipeline elements
// ------------------------------------------------------------------------
void if_process (uint key, uint payload)
{
#ifdef DEBUG
  recv_fwd++;
  if (phase != SPINN_FORWARD)
    wrng_phs++;
#endif

  // get net index: mask out block, phase and colour data,
  uint inx = key & SPINN_NET_MASK;

  // store received net to be processed,
  i_nets[inx] = (long_net_t) ((net_t) payload);

  net_t net_tmp;

  // compute unit input,
  //TODO: need to make sure this is the same as Lens
  compute_in (inx);

  // saturate and cast the long nets before sending,
  if (i_nets[inx] >= (long_net_t) SPINN_NET_MAX)
  {
    net_tmp = (net_t) SPINN_NET_MAX;
  }
  else if (i_nets[inx] <= (long_net_t) SPINN_NET_MIN)
  {
    net_tmp = (net_t) SPINN_NET_MIN;
  }
  else
  {
    net_tmp = (net_t) i_nets[inx];
  }

  // incorporate net index to the packet key and send,
  while (!spin1_send_mc_packet ((fwdKey | inx), net_tmp, WITH_PAYLOAD));

#ifdef DEBUG
  pkt_sent++;
  sent_fwd++;
#endif

  // mark net as done,
  if_done++;

  // and check if all nets done
  if (if_done == icfg.num_units)
  {
    // prepare for next tick,
    if_done = 0;

    // access thread semaphore with interrupts disabled
    uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(if_thrds_pend & SPINN_THRD_PROC))
      wrng_pth++;
#endif

    // check if all other threads done
    if (if_thrds_pend == SPINN_THRD_PROC)
    {
      // if done initialise semaphore,
      if_thrds_pend = SPINN_IF_THRDS;

      // restore interrupts after flag access,
      spin1_mode_restore (cpsr);

      // and advance tick
      if_advance_tick ();
    }
    else
    {
      // if not done report processing thread done,
      if_thrds_pend &= ~SPINN_THRD_PROC;

      // and restore interrupts after flag access
      spin1_mode_restore (cpsr);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP phase: apply BACKPROP input pipeline elements
// ------------------------------------------------------------------------
void ib_process (uint key, uint payload)
{
#ifdef DEBUG
  recv_bkp++;
  if (phase != SPINN_BACKPROP)
    wrng_phs++;
#endif

  // get delta index: mask out block, phase and colour data,
  uint inx = key & SPINN_DELTA_MASK;

  // store received delta to be processed,
  i_deltas[inx] = ((long_delta_t) ((delta_t) payload))
    << (SPINN_LONG_DELTA_SHIFT - SPINN_DELTA_SHIFT);

  // restore net for the previous tick
  restore_net (inx, tick - 1);

  compute_in_back (inx);

  // saturate and cast the long deltas before sending
  long_delta_t delta_tmp = i_deltas[inx]
                         >> (SPINN_LONG_DELTA_SHIFT - SPINN_DELTA_SHIFT);
  delta_t delta;

  if (delta_tmp >= (long_delta_t) SPINN_DELTA_MAX)
  {
    delta = (delta_t) SPINN_DELTA_MAX;
  }
  else if (delta_tmp <= (long_delta_t) SPINN_DELTA_MIN)
  {
    delta = (delta_t) SPINN_DELTA_MIN;
  }
  else
  {
    delta = (delta_t) delta_tmp;
  }

  // incorporate delta index to the packet key and send,
  while (!spin1_send_mc_packet ((i_bkpKey[inx >> SPINN_BLOCK_SHIFT] | inx), delta, WITH_PAYLOAD));

#ifdef DEBUG
  pkt_sent++;
  sent_bkp++;
#endif

  // mark delta as done,
  ib_done++;

  // and check if all deltas done
  if (ib_done == icfg.num_units)
  {
    // prepare for next tick,
    ib_done = 0;

    // and advance tick
    ib_advance_tick ();
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: the tick has been completed, move FORWARD to the next tick
// updating the indices to the events/examples as required
// ------------------------------------------------------------------------
void if_advance_tick (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "if_advance_tick\n");
#endif

#ifdef DEBUG
  tot_tick++;
#endif

  // check if end of event
  if (tick_stop)
  {
    if_advance_event ();
  }
  else
  {
    // if not done increment tick
    tick++;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: the tick has been completed, move FORWARD to the next tick
// updating the indices to the events/examples as required
// ------------------------------------------------------------------------
void ib_advance_tick (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "ib_advance_tick\n");
#endif

#ifdef DEBUG
  tot_tick++;
#endif

  // check if end of BACKPROP phase
  if (tick == SPINN_IB_END_TICK)
  {
    // initialise the tick count
    tick = SPINN_I_INIT_TICK;

    // switch to FORWARD phase,
    phase = SPINN_FORWARD;

    // and move to next example
    i_advance_example ();
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
void if_advance_event (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "if_advance_event\n");
#endif

  // check if done with example's FORWARD phase
  if ((++evt >= num_events) || (tick == ncfg.global_max_ticks - 1))
  {
    // and check if in training mode
    if (xcfg.training)
    {
       // move on to BACKPROP phase
      phase = SPINN_BACKPROP;
    }
    else
    {
      // if not training, initialise ticks for the next example
      tick = SPINN_I_INIT_TICK;

      // then move to next example
      i_advance_example ();
    }
  }
  else
  {
    // if input or output group update input/target index
    //TODO: check if the target value is required in I cores
    // for the BACKPROP phase, otherwise remove the condition for the
    // output group
    if (icfg.input_grp || icfg.output_grp)
    {
      i_it_idx += icfg.num_units;
    }

    // and increment tick
    tick++;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// update example at the end of a (FORWARD or BACKPROP) tick
// ------------------------------------------------------------------------
void i_advance_example (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "i_advance_example\n");
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

      // restore interrupts after flag access,
      spin1_mode_restore (cpsr);

      // and decide what to do
      if (net_stop)
      {
        // finish stage and report no error
        //TODO: check if need to schedule or can simply call
        spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
      }
    }
    else
    {
      // flag ready for net_stop decision,
      net_stop_rdy = TRUE;

      // and restore interrupts after flag access
      spin1_mode_restore (cpsr);
    }

    // and reset example count for next epoch
    example_cnt = 0;
  }

  // start from first event for next example,
  evt = 0;
  num_events = ex[example_inx].num_events;
  event_idx = ex[example_inx].ev_idx;

  // and initialise event input and target indices - if input or output group
  //TODO: check if the target value is required in I cores
  // for the BACKPROP phase, otherwise remove condition for output group
  if (icfg.input_grp || icfg.output_grp)
  {
    i_it_idx = ev[event_idx].it_idx * icfg.num_units;
  }

  // if the input INTEGRATOR is used reset the array of last values
  if (icfg.in_integr_en)
    for (uint i = 0; i < icfg.num_units; i++)
    {
      i_last_integr_net[i] = (long_net_t) icfg.initNets;
      i_last_integr_delta[i] = 0;
    }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase:
// call the elements in the input pipeline
// ------------------------------------------------------------------------
void compute_in (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "compute_in\n");
#endif

  for (uint i = 0; i < icfg.num_in_procs; i++)
  {
    i_in_procs[icfg.procs_list[i]] (inx);
  }

  // check if in training mode, and if so, store nets
  //TODO: for non-continuous networks, this needs to check the requirement
  // to have these histories saved, which needs to come as a configuration
  // parameter. For continuous networks, these histories are always required.
  if (xcfg.training)
  {
    store_net(inx);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// input INTEGRATOR element
// ------------------------------------------------------------------------
void in_integr (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "in_integr\n");
#endif

  long_net_t  last_net = i_last_integr_net[inx];
  long_net_t  desired_net = i_nets[inx];
  long_fpreal dt = icfg.in_integr_dt;

  // compute the new value of the net as indicated by lens
  // all the variables are expanded to long types to avoid overflows and wrap-around
  long_net_t net = last_net + (dt * (desired_net - last_net) >> SPINN_LONG_FPREAL_SHIFT);

  // saturate the value computed and assign it to the nets variable
  // to be used in the next stage of computation
  if (net > (long_net_t) SPINN_NET_MAX)
    i_nets[inx] = (long_net_t) SPINN_NET_MAX;
  else if (net < (long_net_t) SPINN_NET_MIN)
    i_nets[inx] = (long_net_t) SPINN_NET_MIN;
  else
    i_nets[inx] = (long_net_t) net;

  // store the outcome of the computation for the next tick
  i_last_integr_net[inx] = i_nets[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
//soft clamp element
// ------------------------------------------------------------------------
void in_soft_clamp (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "in_soft_clamp\n");
#endif

  // compute only if input is not NaN
  if (it[i_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    long_activ_t external_input = it[i_it_idx + inx];

    long_fpreal soft_clamp_strength = icfg.soft_clamp_strength;

    long_activ_t init_output = icfg.initOutput;

    // computation of the soft clamp operator following Lens code
    long_activ_t output = init_output
                             + ((soft_clamp_strength
                                 * (external_input - init_output))
                                   >> SPINN_FPREAL_SHIFT
                               );

    i_nets[inx] += inv_sigmoid((short_activ_t) (output << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)));
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// routine which computes the BACKPROP phase of the computation of the
// input elements pipeline
// ------------------------------------------------------------------------
void compute_in_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "compute_in_back\n");
#endif

  // the set of procedures needs to be executed in the reverse order, starting
  // from the last input pipeline element, and executing the routine only if the
  // element in the list is not NULL
  if (icfg.num_in_procs >= 1)
  {
    for (int i = icfg.num_in_procs - 1; i >= 0; i--)
    {
      if (i_in_back_procs[icfg.procs_list[i]] != NULL)
        i_in_back_procs[icfg.procs_list[i]] (inx);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the input integration operation for the BACKPROP phase
// ------------------------------------------------------------------------
void in_integr_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "in_integr_back\n");
#endif

  long_delta_t last_delta = i_last_integr_delta[inx];

  long_fpreal dt = icfg.in_integr_dt;

  long_delta_t d = (dt * last_delta) >> SPINN_FPREAL_SHIFT;

  last_delta += i_deltas[inx] - d;

  i_deltas[inx] = d;

  // store the INTEGRATOR state for the next iteration
  i_last_integr_delta[inx] = last_delta;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
/* There is no softClampInputBack in Lens*/
/*
void in_soft_clamp_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "in_soft_clamp_back\n");
#endif
}
*/
// ------------------------------------------------------------------------
