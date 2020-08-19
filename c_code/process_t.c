// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "mlp_externs.h"

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"
#include "activation.h"


// ------------------------------------------------------------------------
// threshold core computation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process FORWARD data packet
// compute a unit output from received net
// ------------------------------------------------------------------------
void tf_process (uint key, uint payload)
{
#ifdef TRACE
  io_printf (IO_BUF, "tf_process\n");
#endif

#ifdef DEBUG
  recv_fwd++;
  if (phase == SPINN_BACKPROP)
    wrng_phs++;
#endif

  // get net index: mask out block, phase and colour data,
  uint inx = (key & SPINN_NET_MASK);

  // packet carries a net as payload,
  t_nets[inx] = (net_t) payload;

  // store net for BACKPROP computation,
  if (xcfg.training)
  {
    store_net (inx);
  }

  // compute unit output,
  //TODO: need to make sure this is the same as Lens
  compute_out (inx);

  // store output for BACKPROP computation,
  if (xcfg.training)
  {
    store_output (inx);
  }

  // send newly computed output to w cores,
  while (!spin1_send_mc_packet ((t_fwdKey[inx >> SPINN_BLOCK_SHIFT] | inx),
                                 (uint) t_outputs[inx],
                                 WITH_PAYLOAD
                               )
        );

#ifdef DEBUG
  pkt_sent++;
  sent_fwd++;
#endif

  // evaluate stop criterion,
  if (tcfg.output_grp)
    tf_stop_func (inx);

  // mark net as arrived,
  tf_arrived++;

  // and check if all nets arrived (i.e., all outputs done)
  if (tf_arrived == tcfg.num_units)
  {
    // initialise scoreboard for next tick,
    tf_arrived = 0;

    // record outputs if recording all ticks,
    if (tcfg.write_out && !tcfg.last_tick_only)
    {
      record_outputs ();
    }

    // access thread semaphore and flags with interrupts disabled,
    uint cpsr = spin1_int_disable ();

    // and check if all other threads done
    if (tcfg.output_grp)
    {
      // report processing thread done,
      //NOTE: tick stop decision cannot have arrived!
      tf_thrds_pend &= ~SPINN_THRD_PROC;

      // check if criterion value can be forwarded
      if (tf_crit_rdy)
      {
        // initialise semaphore,
        tf_crit_rdy = tf_init_crit;

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // send (criterion/tick stop) packet,
        tf_send_stop ();

        // and advance tick if last group
        //NOTE: last group does not get a stop decision
        if (tcfg.is_last_output_group)
        {
          //TODO: check if need to schedule or can simply call
          tf_advance_tick ();
        }
      }
      else
      {
        // flag that local value is ready,
        tf_crit_rdy = 1;

        // and restore interrupts after flag access
        spin1_mode_restore (cpsr);
      }
    }
    else
    {
      // check if all other threads done
      if (tf_thrds_pend == SPINN_THRD_PROC)
      {
        // initialise semaphore,
        tf_thrds_pend = SPINN_TF_THRDS;

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        //TODO: check if need to schedule or can simply call
        tf_advance_tick ();
      }
      else
      {
        // if not done report processing thread done,
        tf_thrds_pend &= ~SPINN_THRD_PROC;

        // and restore interrupts after flag access
        spin1_mode_restore (cpsr);
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP-phase tick
// compute error deltas
// ------------------------------------------------------------------------
void tb_process (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "tb_process\n");
#endif

  // compute deltas based on pre-computed errors,
  //TODO: this needs checking!
  for (uint inx = 0; inx < tcfg.num_units; inx++)
  {
    if (tcfg.output_grp)
    {
      // output groups:
      // restore output derivative for the current tick
      restore_output_deriv (inx, tick);

      // inject error derivative!
      t_output_deriv[inx] += ((long_deriv_t) t_errors[tb_procs][inx])
                               << (SPINN_LONG_DERIV_SHIFT - SPINN_ERROR_SHIFT);
    }
    else
    {
      // non-output groups:
      // use received error computed in previous tick
      t_output_deriv[inx] = ((long_deriv_t) t_errors[tb_procs][inx])
                               << (SPINN_LONG_DERIV_SHIFT - SPINN_ERROR_SHIFT);
    }

    // restore net for the current tick
    restore_net (inx, tick);

    compute_out_back (inx);

    delta_t delta = t_deltas[inx];

    // restore output for the previous forward tick
    restore_output (inx, tick - 1);

    // send delta to input core for further processing
    while (!spin1_send_mc_packet ((bkpKey | inx), (uint) delta, WITH_PAYLOAD));

#ifdef DEBUG
    pkt_sent++;
    sent_bkp++;
#endif
  }

  // access thread semaphore with interrupts disabled
  uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
  if (!(tb_thrds_pend & SPINN_THRD_PROC))
    wrng_pth++;
#endif

  // and check if all other threads done
  if (tb_thrds_pend == SPINN_THRD_PROC)
  {
    // if done initialise thread semaphore,
    tb_thrds_pend = SPINN_TB_THRDS;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // and advance tick
    tb_advance_tick (0, 0);
  }
  else
  {
    // if not done report processing thread done,
    tb_thrds_pend &= ~SPINN_THRD_PROC;

    // and restore interrupts after flag access
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void tf_advance_tick (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "tf_advance_tick\n");
#endif

#ifdef DEBUG
  tot_tick++;
#endif

  // check if done with event
  if (tick_stop)
  {
    // update event criterion
    if (tcfg.is_last_output_group)
    {
      tf_event_crit = tf_event_crit && tf_group_crit && (ev_tick >= min_ticks);
      max_evt = evt;
    }

    // and move to next event
    tf_advance_event ();
  }
  else
  {
    // if not done increment ticks
    tick++;
    ev_tick++;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void tb_advance_tick (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "tb_advance_tick\n");
#endif

#ifdef DEBUG
  tot_tick++;
#endif

  // update pointer to processing unit outputs,
  tb_procs = 1 - tb_procs;

  // check if done with BACKPROP phase
  if (tick == SPINN_TB_END_TICK)
  {
    // initialise the tick count
    tick = SPINN_T_INIT_TICK;

    // initialise the event tick count
    ev_tick = SPINN_T_INIT_TICK;

    // move on to FORWARD phase,
    t_switch_to_fw ();

    // update example criterion,
    if (tcfg.is_last_output_group)
    {
      tf_example_crit = tf_example_crit && tf_event_crit && (max_evt >= num_events - 1);
    }

    // and advance to next example,
    t_advance_example ();
  }
  else
  {
    // if not done decrement tick
    tick--;

    // and trigger computation
    spin1_schedule_callback (tb_process, 0, 0, SPINN_TB_PROCESS_P);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: update the event at the end of a simulation tick
// ------------------------------------------------------------------------
void tf_advance_event (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "tf_advance_event\n");
#endif

  // check if done with example's FORWARD phase
  if ((++evt >= num_events) || (tick == ncfg.global_max_ticks - 1))
  {
    // record outputs if only recording last tick,
    if (tcfg.write_out && tcfg.last_tick_only)
    {
      evt--;  // correct event number before recording outputs
      record_outputs ();
    }

    // and check if in training mode
    if (xcfg.training)
    {
      // move on to BACKPROP phase
      t_switch_to_bp ();
    }
    else
    {
      // if not training,
      // add this example to the tally of examples tested for the current stage
      t_test_results.examples_tested += 1;

      // add the number of ticks on this example to the tally of ticks for the current stage
      // [LENS adds one to the tick count for each example for some reason
      // the same is done here for comparison purposes]
      t_test_results.ticks_tested += (tick + 1);

      // initialise ticks for the next example
      tick = SPINN_T_INIT_TICK;
      ev_tick = SPINN_T_INIT_TICK;

      // then advance to next example
      t_advance_example ();
    }
  }
  else
  {
    // if input or output group update parameters
    if (tcfg.input_grp || tcfg.output_grp)
    {
      // update input/target index,
      t_it_idx += tcfg.num_units;

      // and update number of ticks for new event
      if (tcfg.is_last_output_group)
      {
        // maximum
        if (ev[event_idx + evt].max_time != SPINN_FP_NaN)
          max_ticks = (((ev[event_idx + evt].max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
        else
          max_ticks = (((es->max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;

        // minimum
        if (ev[event_idx + evt].min_time != SPINN_FP_NaN)
          min_ticks = (((ev[event_idx + evt].min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
        else
          min_ticks = (((es->min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
      }
    }

    // increment example tick,
    tick++;

    // and initialise event tick
    ev_tick = SPINN_T_INIT_TICK;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// update example at the end of a (FORWARD or BACKPROP) tick
// ------------------------------------------------------------------------
void t_advance_example (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "t_advance_example\n");
#endif

  // network stop decision,
  uchar nsd = 0;

  // point to next example in the set - wrap around if at the end,
  if (++example_inx >= es->num_examples)
  {
    example_inx = 0;
  }

  // check if done with examples,
  //TODO: alternative algorithms for choosing example order!
  if (++example_cnt >= xcfg.num_examples)
  {
    // prepare for next epoch,
    epoch++;

    // check if stage done,
    if (tcfg.is_last_output_group)
    {
      // report network stop decision,
      nsd = (!xcfg.training || (epoch >= xcfg.num_epochs)) ? 1 : tf_example_crit;

      // broadcast network_stop decision,
      while (!spin1_send_mc_packet (tf_stpn_key | nsd,
          0, NO_PAYLOAD)
          );

#ifdef DEBUG
      pkt_sent++;
      stn_sent++;
#endif

      // and finish if done with epochs
      if (nsd)
      {
        // report no error
        spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
      }
    }
    else
    {
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
          // finish stage and report no error
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
    }

    // reset example count for next epoch,
    example_cnt = 0;

    // initialise stopping criteria for next epoch,
    tf_event_crit = 1;
    tf_example_crit = 1;

    // initialise test result variables for next epoch,
    //NOTE: do not initialise if reporting!
    if (!nsd)
    {
      t_test_results.examples_tested = 0;
      t_test_results.ticks_tested = 0;
      t_test_results.examples_correct = 0;
    }

    // and increment the count of epochs trained
    if (xcfg.training)
    {
      t_test_results.epochs_trained++;
    }
  }

  // start from first event for next example,
  evt = 0;
  num_events = ex[example_inx].num_events;
  event_idx = ex[example_inx].ev_idx;
  tf_event_crit = 1;

  // if input or output group initialise new event input/target index,
  if (tcfg.input_grp || tcfg.output_grp)
  {
    t_it_idx = ev[event_idx].it_idx * tcfg.num_units;
  }

  // initialise output function outputs,
  t_init_outputs ();

  // and update next event data
  if (tcfg.is_last_output_group)
  {
    // update number of ticks for new event,
    // maximum
    if (ev[event_idx + evt].max_time != SPINN_FP_NaN)
      max_ticks = (((ev[event_idx + evt].max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;
    else
      max_ticks = (((es->max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;

    // minimum
    if (ev[event_idx + evt].min_time != SPINN_FP_NaN)
      min_ticks = (((ev[event_idx + evt].min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;
    else
      min_ticks = (((es->min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: when the simulation is completed in the BACKPROP phase,
// switch to the FORWARD phase again, if required
// ------------------------------------------------------------------------
void t_switch_to_fw (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "t_switch_to_fw\n");
#endif

  // move to new FORWARD phase,
  phase = SPINN_FORWARD;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: when the simulation is completed in the FORWARD phase,
// switch to the backward phase if training is required
// ------------------------------------------------------------------------
void t_switch_to_bp (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "t_switch_to_bp\n");
#endif

  // move to new BACKPROP phase,
  phase = SPINN_BACKPROP;

  // initialise t_errors for next example,
  for (uint i = 0; i < tcfg.num_units; i++)
  {
    t_errors[tb_procs][i] = 0;
  }

  // and trigger BACKPROP computation
  spin1_schedule_callback (tb_process, 0, 0, SPINN_TB_PROCESS_P);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// this routine calls, in the appropriate order, all the elements of the
// output pipeline, as expressed in array tcfg.procs_list.
// the routines to be called are listed in array t_out_procs.
// the return value of the error function (output_deriv) needs to
// be stored to be used in the BACKPROP phase.
// the network type has implications for the initialisation routines,
// as the memory where to store histories needs to be allocated in SDRAM.
// ------------------------------------------------------------------------
void compute_out (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "compute_out\n");
#endif

  // initialise the array element where to store the output value for the
  t_outputs[inx] = 0;

  // compute all the elements of the output pipeline
  // from the observations in lens, the logistic is always the first element of
  // the output pipeline, which uses the value received through the multicast
  // packet. If no logistic function is used, the t_outputs starts with a 0
  // value, as initialised earlier
  for (uint i = 0; i < tcfg.num_out_procs; i++)
  {
    t_out_procs[tcfg.procs_list[i]] (inx);
  }

  // if the network is set for training, then compute the output derivative
  // using the appropriate function
  if (xcfg.training && tcfg.output_grp)
  {
    // if the error function to be called is not NULL,
    // compute the output derivative
    if (t_out_error[tcfg.error_function] != NULL)
    {
      t_out_error[tcfg.error_function] (inx);
    }
  }

  // if in training mode store targets and output derivatives.
  //TODO: for non-continuous networks, this needs to check the requirement
  //TODO: to have these histories saved, which needs configuration parameter.
  //TODO: For continuous networks, these are always required.
  if (xcfg.training)
  {
    store_output_deriv (inx);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the logistic function starting from the value received through the
// multicast packet.
// ------------------------------------------------------------------------
void out_logistic (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_logistic\n");
#endif

  // compute the sigmoid using a lookup table and an interpolation function
  t_outputs[inx] = sigmoid (t_nets[inx]);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the output integration operation
// ------------------------------------------------------------------------
void out_integr (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_integr\n");
#endif

  activation_t last_output = t_last_integr_output[inx];

  activation_t new_output = t_outputs[inx];

  long_fpreal dt = tcfg.out_integr_dt;


  // store the output for the backward path
  t_instant_outputs[((tick - 1) * tcfg.num_units) + inx] = t_outputs[inx];

  // compute the of the output INTEGRATOR and round off
  long_activ_t out_tmp = ((dt * (new_output - last_output))
                            + (1 << SPINN_ACTIV_SHIFT))
                            >> SPINN_FPREAL_SHIFT;

  out_tmp += last_output;

  // saturate the value computed and assign it to the output variable
  if (out_tmp > (long_activ_t) (SPINN_SHORT_ACTIV_MAX << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)))
    // positive saturation
    t_outputs[inx] = (activation_t) (SPINN_SHORT_ACTIV_MAX << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT));
  else if (out_tmp < (long_activ_t) (SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)))
    // negative saturation
    t_outputs[inx] = (activation_t) (SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT));
  else
    // no saturation needed
    t_outputs[inx] = (activation_t) out_tmp;

  // store the INTEGRATOR state for the next iteration
  t_last_integr_output[inx] = t_outputs[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the output hard clamp, and store the injected value in SDRAM
// to be used in the BACKPROP phase: in fact, the BACKPROP phase needs to know
// to which units a value has been injected so that the output derivative
// in the BACKPROP phase can be set appropriately
// ------------------------------------------------------------------------
void out_hard_clamp (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_hard_clamp\n");
#endif

  // compute only if input is not NaN
  if (it[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    // assign the value coming from the event
    t_outputs[inx] = it[t_it_idx + inx];
  }

  //TODO: if training, store the injected value in SDRAM. This memory area needs
  // to be allocated during initialisation
/*
  if (xcfg.training)
  {
    short_activ_t * tmp = t_out_hard_clamp_data + tick * tcfg.num_units;
    tmp[inx] = t_outputs[inx];
  }
*/
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the bias clamp. This clamp is used only by the bias group, and sets
// the output value to a constant maximum output value of 1
// ------------------------------------------------------------------------
void out_bias (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_bias\n");
#endif

  // set output value to 1
  t_outputs[inx] = SPINN_ACTIV_ONE;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the weak clamp, as defined by lens, and store the injected value, if
// the network needs training
// ------------------------------------------------------------------------
void out_weak_clamp (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_weak_clamp\n");
#endif

  //TODO: if training, store the injected value in SDRAM. This memory area needs
  // to be allocated during initialisation
/*
  if (xcfg.training)
  {
    //store previous value of t_output for BACKPROP computation
    short_activ_t * tmp = t_out_weak_clamp_data + tick * tcfg.num_units;
    tmp[inx] = t_outputs[inx] << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT);
  }
*/

  // compute only if input is not NaN
  if (it[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    long_activ_t external_input = it[t_it_idx + inx];
    long_fpreal weak_clamp_strength = tcfg.weak_clamp_strength;
    long_activ_t output_value = t_outputs[inx];

    // computation of the weak clamp output following Lens implementation
    long_activ_t output = output_value
                             + ((weak_clamp_strength
                                 * (external_input - output_value))
                                   >> SPINN_FPREAL_SHIFT
                               );

    // saturate and cast output
    if (output > (long_activ_t) (SPINN_SHORT_ACTIV_MAX << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)))
      t_outputs[inx] = (activation_t) SPINN_SHORT_ACTIV_MAX << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT);
    else if (output < (long_activ_t) (SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)))
      t_outputs[inx] = (activation_t) SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT);
    else
      t_outputs[inx] = (activation_t) output;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// routine to compute the BACKPROP phase of the elements of the output pipeline
// the elements need to be computed in the reverse order, if they exist
// ------------------------------------------------------------------------
void compute_out_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "compute_out_back\n");
#endif

  // if the output pipeline includes one or more elements, compute them in the
  // reverse order
  if (tcfg.num_out_procs >= 1)
  {
    for (int i = tcfg.num_out_procs - 1; i >= 0; i--)
    {
      if (t_out_back_procs[tcfg.procs_list[i]] != NULL)
      {
        t_out_back_procs[tcfg.procs_list[i]] (inx);
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// derivative of the logistic function computed through a lookup table
// ------------------------------------------------------------------------
void out_logistic_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_logistic_back\n");
#endif

  // compute output * (1 - output),
  long_activ_t tmp1 = (long_activ_t) t_outputs[inx] * (long_activ_t) ((1 << SPINN_ACTIV_SHIFT) - t_outputs[inx]);

  // round off
  tmp1 += 1 << (SPINN_ACTIV_SHIFT + SPINN_ACTIV_SHIFT - SPINN_ACTIV_SHIFT - 1);

  // adjust decimal point position
  tmp1 = (tmp1 >> (SPINN_ACTIV_SHIFT + SPINN_ACTIV_SHIFT - SPINN_ACTIV_SHIFT));

  // compute error delta,
  long_delta_t tmp2 = (long_delta_t) t_output_deriv[inx] * tmp1;

  // round off,
  tmp2 += 1 << (SPINN_LONG_DERIV_SHIFT + SPINN_ACTIV_SHIFT - SPINN_DELTA_SHIFT - 1);

  t_deltas[inx] = (delta_t) (tmp2 >> (SPINN_LONG_DERIV_SHIFT
                              + SPINN_ACTIV_SHIFT - SPINN_DELTA_SHIFT));
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the output integration operation for the BACKPROP phase
// ------------------------------------------------------------------------
void out_integr_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_integr_back\n");
#endif

  long_deriv_t last_output_deriv = t_last_integr_output_deriv[inx];

  long_fpreal dt = (long_fpreal) tcfg.out_integr_dt;

  // reset output to value stored during forward pass
  t_outputs[inx] = t_instant_outputs[((tick - 1) * tcfg.num_units) + inx];

  long_deriv_t d = (dt * last_output_deriv) >> SPINN_FPREAL_SHIFT;
  last_output_deriv += t_output_deriv[inx] - d;
  t_output_deriv[inx] = d;

  // store the INTEGRATOR state for the next iteration
  t_last_integr_output_deriv[inx] = last_output_deriv;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase for the hard clam
//TODO: this is a stub
// ------------------------------------------------------------------------
void out_hard_clamp_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_hard_clamp_back\n");
#endif

  (void) inx;

/*
  short_activ_t * tmp = t_out_hard_clamp_data + tick * tcfg.num_units;

  if (tmp[inx] != SPINN_SHORT_ACTIV_NaN)
    t_output_deriv[inx] = 0;
*/
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase for the weak clamp
//TODO: this is a stub
// ------------------------------------------------------------------------
void out_weak_clamp_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_weak_clamp_back\n");
#endif

  (void) inx;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
//BACKPROP phase for the bias clamp
// ------------------------------------------------------------------------
void out_bias_back (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "out_bias_back\n");
#endif

  t_output_deriv[inx] = 0;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// evaluation of the standard convergence criterion
// for each unit in the output group check if the output value is close
// to the target value, with the "group_criterion" variable defining the
// acceptance margin.
// ------------------------------------------------------------------------
void std_stop_crit (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "std_stop_crit\n");
#endif

  // evaluate only if target is not NaN
  if (tt[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    error_t error = (error_t) ABS ((t_outputs[inx] - tt[t_it_idx + inx]) >>
                (SPINN_ACTIV_SHIFT - SPINN_ERROR_SHIFT));

    tf_stop_crit = tf_stop_crit && (error < t_group_criterion);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// evaluation of the "max" convergence criterion
// for each unit in the output group check if both the output and target values
// are the maximum in the group and, in this case, if their difference is less
// or equal than the tcfg.group_criterion value.
//TODO: this routine needs to be modified to adapt to the case in which the
// output group is split across multiple cores, as this is a global convergence
// rule, rather than an individual one, as the standard convergence criterion.
// ------------------------------------------------------------------------
void max_stop_crit (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "max_stop_crit\n");
#endif

  // evaluate only if target is not NaN
  if (tt[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    if (t_outputs[inx] > t_max_output)
    {
      t_max_output = t_outputs[inx];
      t_max_output_unit = inx;
    }

    if (tt[t_it_idx + inx] > t_max_target)
    {
      t_max_target = tt[t_it_idx + inx];
      t_max_target_unit = inx;
    }

    error_t error = (error_t) ABS ((t_max_output - t_max_target) >>
                        (SPINN_ACTIV_SHIFT - SPINN_ERROR_SHIFT));

    if ((t_max_output_unit == -1)
         || ((t_max_output_unit == t_max_target_unit)
             && (error < t_group_criterion)
            )
       )
      tf_stop_crit = TRUE;
    else
      tf_stop_crit = FALSE;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the output derivative as derivative of the squared error function:
// (output - target) * 2
// ------------------------------------------------------------------------
void error_squared (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "error_squared\n");
#endif

  // evaluate only if target is not NaN
  if (tt[t_it_idx + inx] != SPINN_ACTIV_NaN)
    t_output_deriv[inx] = ((long_deriv_t) t_outputs[inx] - (long_deriv_t) tt[t_it_idx + inx]);
  else
    t_output_deriv[inx] = 0;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// this routine has been extracted and rewritten and tested starting from the
// LENS code above. SMALL_VAL and LARGE_VAL are constants defined by LENS and
// represent a sort of saturation values, since the derivative of the cross
// entropy function has two discontinuities for the output value equal to 0 and
// to 1. The general version of the derivative of the cross entropy function is:
// (output - target) / (output * (1 - output))
// ------------------------------------------------------------------------
void error_cross_entropy (uint inx)
{
#ifdef TRACE
  io_printf (IO_BUF, "error_cross_entropy\n");
#endif

  // if the target is defined, compute the output derivative, otherwise set it to 0
  if (tt[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    // if the target is 0, then the cross entropy function simplifies in
    // 1 / (1 - output)
    if (tt[t_it_idx + inx] == 0)
    {
      // if the output value is close to 1, then the cross entropy function
      // 1 / (1 - output) has a discontinuity and the result is set to the
      // largest value that can be represented
      if ((activation_t) SPINN_ACTIV_ONE - t_outputs[inx] <= (activation_t) (SPINN_SMALL_VAL << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)))
      {
        t_output_deriv[inx] = (long_deriv_t) SPINN_DERIV_MAX;
      }
      // otherwise compute 1 / (1 - output)
      else
      {
        derivative_t numerator = (derivative_t) SPINN_DERIV_ONE;
        derivative_t denominator = (derivative_t) SPINN_DERIV_ONE - (t_outputs[inx] >> (SPINN_ACTIV_SHIFT - SPINN_DERIV_SHIFT));

        // the left shift needs to be done before the division, as
        // precision decreases with the division
        t_output_deriv[inx] = (((long_deriv_t) numerator << SPINN_LONG_DERIV_SHIFT) / (long_deriv_t) denominator);
      }
    }
    // if the target is close to 1, then the cross entropy function simplifies:
    // -1 / output
    else if (tt[t_it_idx + inx] == ((activation_t) SPINN_ACTIV_ONE))
    {
      // if the output value is close to 0, then the cross entropy function
      // shows a discontinuity, and the output value is set to the minimum
      // negative value that can be represented
      if (t_outputs[inx] <= (activation_t) (SPINN_SMALL_VAL << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)))
      {
        t_output_deriv[inx] = (long_deriv_t) SPINN_DERIV_MIN;
      }
      // otherwise compute -1 / output
      else
      {
        derivative_t numerator = (derivative_t) SPINN_DERIV_NEG_ONE;
        derivative_t denominator = (t_outputs[inx] >> (SPINN_ACTIV_SHIFT - SPINN_DERIV_SHIFT));

        // the left shift needs to be done before the division, as the
        // precision reduces with the division
        t_output_deriv[inx] = (((long_deriv_t) numerator << SPINN_LONG_DERIV_SHIFT) / (long_deriv_t) denominator);
      }
    }
    // otherwise compute the standard function
    else
    {
      // if (output * (1-output)) is close to 0, the function presents a
      // discontinuity, and the result is computed as
      // MAX_VALUE * (output - target)
      // where the MAX value is the maximum representable value
      if (( ((long_activ_t) t_outputs[inx] * (long_activ_t) ((activation_t) SPINN_ACTIV_ONE - t_outputs[inx])) << (long_activ_t) SPINN_ACTIV_SHIFT) <= (activation_t) SPINN_SMALL_VAL << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT))
      {
        t_output_deriv[inx] = ((((long_deriv_t) SPINN_DERIV_MAX) * (long_deriv_t)(t_outputs[inx] - tt[t_it_idx + inx])) >> SPINN_ACTIV_SHIFT);
      }
      // otherwise compute the standard formula
      // (output - target) / (output * (1 - output))
      else
      {
        derivative_t numerator = ((derivative_t) (t_outputs[inx] >> (SPINN_ACTIV_SHIFT - SPINN_DERIV_SHIFT)) - (derivative_t) (tt[t_it_idx + inx] >> (SPINN_ACTIV_SHIFT - SPINN_DERIV_SHIFT)));
        derivative_t one = (derivative_t) SPINN_DERIV_ONE;
        long_deriv_t denominator = ((long_deriv_t) t_outputs[inx] * (long_deriv_t) (one - (t_outputs[inx] >> (SPINN_ACTIV_SHIFT - SPINN_DERIV_SHIFT)))) >> SPINN_ACTIV_SHIFT;

        t_output_deriv[inx] = (((long_deriv_t) numerator << SPINN_LONG_DERIV_SHIFT) / (long_deriv_t) denominator);
      }
    }
  }
  // if the target is not defined, set the output derivative to 0
  else
  {
    t_output_deriv[inx] = 0;
  }
}
// ------------------------------------------------------------------------
