/*
 * Copyright (c) 2015 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "mlp_externs.h"

#include "init_w.h"
#include "comms_w.h"
#include "process_w.h"
#include "activation.h"


// ------------------------------------------------------------------------
// weight core computation routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// process a FORWARD-phase tick
// compute partial dot products (output * weight)
// ------------------------------------------------------------------------
void wf_process (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "wf_process\n");
#endif

  // compute all net block dot-products and send them for accumulation,
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
#ifdef PROFILE
    // start profiler
    tc[T2_LOAD] = SPINN_PROFILER_START;
#endif

    long_net_t net_part_tmp = 0;

    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      net_part_tmp += (((long_net_t) w_outputs[wf_procs][i] * (long_net_t) w_weights[i][j])
                  >> (SPINN_ACTIV_SHIFT + SPINN_WEIGHT_SHIFT - SPINN_LONG_NET_SHIFT));
    }

    net_t net_part = 0;

    // saturate the value computed and assign it to the net_part variable
    if (net_part_tmp > (long_net_t) SPINN_NET_MAX)
      // positive saturation
      net_part = (net_t) SPINN_NET_MAX;
    else if (net_part_tmp < (long_net_t) SPINN_NET_MIN)
      // negative saturation
      net_part = (net_t) SPINN_NET_MIN;
    else
      // no saturation needed
      net_part = (net_t) net_part_tmp;

    // incorporate net index to the packet key and send
    while (!spin1_send_mc_packet ((fwdKey | j), (uint) net_part, WITH_PAYLOAD));

#ifdef DEBUG
    sent_fwd++;
#endif

#ifdef PROFILE
    // update profiler values
    uint cnt = SPINN_PROFILER_START - tc[T2_COUNT];
    if (cnt < prf_fwd_min) prf_fwd_min = cnt;
    if (cnt > prf_fwd_max) prf_fwd_max = cnt;
#endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP data packet
// compute partial products (weight * delta)
// ------------------------------------------------------------------------
void wb_process (uint key, uint payload)
{
#ifdef DEBUG
  recv_bkp++;
  if (phase == SPINN_FORWARD)
    wrng_bph++;
#endif

#ifdef PROFILE
  // start profiler
  tc[T2_LOAD] = SPINN_PROFILER_START;
#endif

  // get delta index: mask out phase and block data,
  uint inx = key & SPINN_DELTA_MASK;

  // packet carries a delta as payload
  delta_t delta = (delta_t) payload;

  // update scoreboard,
  wb_arrived++;

  // partial value used to compute Doug's Momentum
  long_lds_t link_delta_sum = 0;

  // compute link derivatives and partial error dot products,
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    // compute link derivatives,
    w_link_deltas[i][inx] += ((long_delta_t) w_outputs[0][i]
                               * (long_delta_t) delta)
                               >> (SPINN_ACTIV_SHIFT + SPINN_DELTA_SHIFT
                               - SPINN_LONG_DELTA_SHIFT);

    // if using Doug's Momentum and reached the end of an epoch
    // accumulate partial link delta sum (to send to s core),
    if (xcfg.update_function == SPINN_DOUGSMOMENTUM_UPDATE
          && example_cnt == (xcfg.num_examples - 1)
          && tick == SPINN_WB_END_TICK)
    {
      // only use link derivatives for links whose weights are non-zero
      // as zero weights indicate no connection
      if (w_weights[i][inx] != 0)
      {
        long_lds_t link_delta_tmp;

        // scale the link derivatives
        if (ncfg.net_type == SPINN_NET_CONT)
        {
          link_delta_tmp = (w_link_deltas[i][inx] * (long_delta_t) w_delta_dt)
                               >> (SPINN_LONG_DELTA_SHIFT + SPINN_FPREAL_SHIFT
                                   - SPINN_LONG_LDS_SHIFT);
        }
        else
        {
          link_delta_tmp = w_link_deltas[i][inx];
        }

        // square the link derivatives
        link_delta_tmp = ((link_delta_tmp * link_delta_tmp) >> SPINN_LONG_LDS_SHIFT);
        link_delta_sum = link_delta_sum + link_delta_tmp;
      }
    }

    // partially compute error dot products,
    //NOTE: may need to make w_errors a long_error_t type and saturate!
    w_errors[i] += (error_t) (((long_error_t) w_weights[i][inx]
                     * (long_error_t) delta)
                     >> (SPINN_WEIGHT_SHIFT + SPINN_DELTA_SHIFT
                     - SPINN_ERROR_SHIFT)
                   );

    // check if done with all deltas
    if (wb_arrived == wcfg.num_cols)
    {
      // send computed error dot product,
      while (!spin1_send_mc_packet ((bkpKey | i), (uint) w_errors[i], WITH_PAYLOAD));

#ifdef DEBUG
      sent_bkp++;
#endif
    }
  }

  // if using Doug's Momentum and reached the end of an epoch,
  // forward the accumulated partial link delta sums to the s core
  if (xcfg.update_function == SPINN_DOUGSMOMENTUM_UPDATE
          && example_cnt == (xcfg.num_examples - 1)
          && tick == SPINN_WB_END_TICK)
  {
    // cast link_delta_sum to send as payload,
    //NOTE: link deltas are unsigned!
    lds_t lds_to_send;

    if (link_delta_sum > (long_lds_t) SPINN_LDS_MAX)
      // positive saturation
      lds_to_send = (lds_t) SPINN_LDS_MAX;
    else
      // no saturation needed
      lds_to_send = (lds_t) link_delta_sum;

    // and send partial link delta sum
    while (!spin1_send_mc_packet (ldsKey, (uint) lds_to_send, WITH_PAYLOAD));

#ifdef DEBUG
    lds_sent++;
#endif
  }

#ifdef PROFILE
  // update profiler values
  uint cnt = SPINN_PROFILER_START - tc[T2_COUNT];
  if (cnt < prf_bkp_min) prf_bkp_min = cnt;
  if (cnt > prf_bkp_max) prf_bkp_max = cnt;
#endif
}
// ------------------------------------------------------------------------+


// ------------------------------------------------------------------------
// perform a weight update using steepest descent
// a weight of 0 means that there is no connection between the two units.
// the zero value is represented by the lowest possible (positive or negative)
// weight.
// ------------------------------------------------------------------------
void steepest_update_weights (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "steepest_update_weights\n");
#endif

#ifdef DEBUG
  wght_ups++;
#endif

  // update weights
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      // do not update weights that are 0 -- indicates no connection!
      if (w_weights[i][j] != 0)
      {
        // scale the link derivatives
        if (ncfg.net_type == SPINN_NET_CONT)
        {
          w_link_deltas[i][j] = (w_link_deltas[i][j]
                                 * (long_delta_t) w_delta_dt)
                                 >> SPINN_FPREAL_SHIFT;
        }

        // compute weight change,
        long_wchange_t change_tmp = ((long_wchange_t) -wcfg.learningRate *
                             (long_wchange_t) w_link_deltas[i][j]);

        // round off,
        change_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                        + SPINN_LONG_DELTA_SHIFT
                                        - SPINN_WEIGHT_SHIFT - 1));

        // and adjust decimal point position
        w_wchanges[i][j] = change_tmp
                             >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_LONG_DELTA_SHIFT
                             - SPINN_WEIGHT_SHIFT);

        if (wcfg.weightDecay > 0)
        {
          //apply weight decay
          long_wchange_t weightDecay_tmp = wcfg.weightDecay * w_weights[i][j];

          // round off
          weightDecay_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                               + SPINN_WEIGHT_SHIFT
                                               - SPINN_WEIGHT_SHIFT - 1));

          // and adjust decimal point position
          weightDecay_tmp = weightDecay_tmp
                             >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_WEIGHT_SHIFT
                             - SPINN_WEIGHT_SHIFT);

          w_wchanges[i][j] = w_wchanges[i][j] - weightDecay_tmp;
        }

        // compute new weight
        long_weight_t temp = (long_weight_t) w_weights[i][j]
                              + (long_weight_t) w_wchanges[i][j];

        // saturate new weight,
        if (temp >= (long_weight_t) SPINN_WEIGHT_MAX)
        {
          w_weights[i][j] = SPINN_WEIGHT_MAX;
        }
        else if (temp <= (long_weight_t) SPINN_WEIGHT_MIN)
        {
          w_weights[i][j] = SPINN_WEIGHT_MIN;
        }
        // and avoid (new weight == 0) -- indicates no connection!
        else if (temp == 0)
        {
          if (w_weights[i][j] > 0)
          {
            w_weights[i][j] = SPINN_WEIGHT_POS_EPSILON;
          }
          else
          {
            w_weights[i][j] = SPINN_WEIGHT_NEG_EPSILON;
          }
        }
        else
        {
          w_weights[i][j] = (weight_t) temp;
        }
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// perform a weight update using momentum descent
// a weight of 0 means that there is no connection between the two units.
// the zero value is represented by the lowest possible (positive or negative)
// weight.
// ------------------------------------------------------------------------
void momentum_update_weights (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "momentum_update_weights\n");
#endif

#ifdef DEBUG
  wght_ups++;
#endif

  // update weights
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      // do not update weights that are 0 -- indicates no connection!
      if (w_weights[i][j] != 0)
      {
        // scale the link derivatives
        if (ncfg.net_type == SPINN_NET_CONT)
        {
          w_link_deltas[i][j] = (w_link_deltas[i][j]
                                 * (long_delta_t) w_delta_dt)
                                 >> SPINN_FPREAL_SHIFT;
        }

        // compute weight change,
        long_wchange_t change_tmp = ((long_wchange_t) -wcfg.learningRate *
                             (long_wchange_t) w_link_deltas[i][j]);


        // round off,
        change_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                        + SPINN_LONG_DELTA_SHIFT
                                        - SPINN_WEIGHT_SHIFT - 1));

        // compute momentum factor
        long_wchange_t momentum_tmp = ((long_wchange_t) wcfg.momentum * w_wchanges[i][j]);

        // round off
        momentum_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                          + SPINN_WEIGHT_SHIFT
                                          - SPINN_WEIGHT_SHIFT - 1));

        // compute sum and adjust decimal point position
        w_wchanges[i][j] =
                (change_tmp >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_LONG_DELTA_SHIFT
                              - SPINN_WEIGHT_SHIFT))
              + (momentum_tmp >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_WEIGHT_SHIFT
                              - SPINN_WEIGHT_SHIFT));

        if (wcfg.weightDecay > 0)
        {
          //apply weight decay
          long_wchange_t weightDecay_tmp = wcfg.weightDecay * w_weights[i][j];

          // round off
          weightDecay_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                               + SPINN_WEIGHT_SHIFT
                                               - SPINN_WEIGHT_SHIFT - 1));

          // and adjust decimal point position
          weightDecay_tmp = weightDecay_tmp
                             >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_WEIGHT_SHIFT
                             - SPINN_WEIGHT_SHIFT);

          w_wchanges[i][j] = w_wchanges[i][j] - weightDecay_tmp;
        }

        // compute new weight
        long_weight_t temp = (long_weight_t) w_weights[i][j]
                              + (long_weight_t) w_wchanges[i][j];

        // saturate new weight,
        if (temp >= (long_weight_t) SPINN_WEIGHT_MAX)
        {
          w_weights[i][j] = SPINN_WEIGHT_MAX;
        }
        else if (temp <= (long_weight_t) SPINN_WEIGHT_MIN)
        {
          w_weights[i][j] = SPINN_WEIGHT_MIN;
        }
        // and avoid (new weight == 0) -- indicates no connection!
        else if (temp == 0)
        {
          if (w_weights[i][j] > 0)
          {
            w_weights[i][j] = SPINN_WEIGHT_POS_EPSILON;
          }
          else
          {
            w_weights[i][j] = SPINN_WEIGHT_NEG_EPSILON;
          }
        }
        else
        {
          w_weights[i][j] = (weight_t) temp;
        }
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// perform a weight update using doug's momentum
// a weight of 0 means that there is no connection between the two units.
// the zero value is represented by the lowest possible (positive or negative)
// weight.
// ------------------------------------------------------------------------
void dougsmomentum_update_weights (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "dougsmomentum_update_weights\n");
#endif

#ifdef DEBUG
  wght_ups++;
#endif

  wchange_t scale;

  if (w_lds_final > SPINN_LDS_ONE)
  {
    // calculate scale = 1/sqrt(w_lds_final)
    wchange_t w_lds_sqrt = sqrt_custom(w_lds_final);
    // s16.15 = (s16.15 << 15) / s16.15
    scale = ((SPINN_WEIGHT_ONE << SPINN_WEIGHT_SHIFT)/w_lds_sqrt);
  }
  else
  {
    scale = SPINN_WEIGHT_ONE;
  }

  // multiply learning scale by learning rate
  scale = (scale * wcfg.learningRate) >> SPINN_SHORT_FPREAL_SHIFT;

  // update weights
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      // do not update weights that are 0 -- indicates no connection!
      if (w_weights[i][j] != 0)
      {
        // scale the link derivatives
        if (ncfg.net_type == SPINN_NET_CONT)
        {
          w_link_deltas[i][j] = (w_link_deltas[i][j]
                                 * (long_delta_t) w_delta_dt)
                                 >> SPINN_FPREAL_SHIFT;
        }

        // compute weight change,
        long_wchange_t change_tmp = ((long_wchange_t) -scale *
                             (long_wchange_t) w_link_deltas[i][j]);


        // round off,
        change_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                        + SPINN_LONG_DELTA_SHIFT
                                        - SPINN_WEIGHT_SHIFT - 1));

        // compute momentum factor
        long_wchange_t momentum_tmp = ((long_wchange_t) wcfg.momentum * w_wchanges[i][j]);

        // round off
        momentum_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                          + SPINN_WEIGHT_SHIFT
                                          - SPINN_WEIGHT_SHIFT - 1));

        // compute sum and adjust decimal point position
        w_wchanges[i][j] =
                (change_tmp >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_LONG_DELTA_SHIFT
                              - SPINN_WEIGHT_SHIFT))
              + (momentum_tmp >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_WEIGHT_SHIFT
                              - SPINN_WEIGHT_SHIFT));

        if (wcfg.weightDecay > 0)
        {
          //apply weight decay
          long_wchange_t weightDecay_tmp = wcfg.weightDecay * w_weights[i][j];

          // round off
          weightDecay_tmp += (long_wchange_t) (1 << (SPINN_SHORT_FPREAL_SHIFT
                                               + SPINN_WEIGHT_SHIFT
                                               - SPINN_WEIGHT_SHIFT - 1));

          // and adjust decimal point position
          weightDecay_tmp = weightDecay_tmp
                             >> (SPINN_SHORT_FPREAL_SHIFT + SPINN_WEIGHT_SHIFT
                             - SPINN_WEIGHT_SHIFT);

          w_wchanges[i][j] = w_wchanges[i][j] - weightDecay_tmp;
        }

        // compute new weight
        long_weight_t temp = (long_weight_t) w_weights[i][j]
                              + (long_weight_t) w_wchanges[i][j];

        // saturate new weight,
        if (temp >= (long_weight_t) SPINN_WEIGHT_MAX)
        {
          w_weights[i][j] = SPINN_WEIGHT_MAX;
        }
        else if (temp <= (long_weight_t) SPINN_WEIGHT_MIN)
        {
          w_weights[i][j] = SPINN_WEIGHT_MIN;
        }
        // and avoid (new weight == 0) -- indicates no connection!
        else if (temp == 0)
        {
          if (w_weights[i][j] > 0)
          {
            w_weights[i][j] = SPINN_WEIGHT_POS_EPSILON;
          }
          else
          {
            w_weights[i][j] = SPINN_WEIGHT_NEG_EPSILON;
          }
        }
        else
        {
          w_weights[i][j] = (weight_t) temp;
        }
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void wf_advance_tick (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "wf_advance_tick\n");
#endif

  // prepare to start tick,
  tick_init (!SPINN_RESTART, 0);

  // and check if end of event
  if (tick_stop)
  {
    wf_advance_event ();
  }
  else
  {
    // if not increment tick,
    tick++;

    // and trigger computation
    spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void wb_advance_tick (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "wb_advance_tick\n");
#endif

  // prepare to start tick,
  tick_init (!SPINN_RESTART, 0);

  // and check if end of example's BACKPROP phase
  if (tick == SPINN_WB_END_TICK)
  {
    // initialise tick for next example
    tick = SPINN_W_INIT_TICK;

    // move on to FORWARD phase,
    w_switch_to_fw ();

    // and move to next example
    w_advance_example ();
  }
  else
  {
    // if not decrement tick,
    tick--;

    // and restore previous tick outputs
    restore_outputs (tick - 1);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: update the event at the end of a simulation tick
// ------------------------------------------------------------------------
void wf_advance_event (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "wf_advance_event\n");
#endif

  // check if done with example's FORWARD phase
  if ((++evt >= num_events) || (tick == ncfg.global_max_ticks - 1))
  {
    // check if in training mode
    if (xcfg.training)
    {
      // move on to BACKPROP phase
      w_switch_to_bp ();
    }
    else
    {
      // if not training initialise tick for next example,
      tick = SPINN_W_INIT_TICK;

      // and move to next example
      w_advance_example ();
    }
  }
  else
  {
    // if not increment tick,
    tick++;

    // and trigger computation
    spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// update example at the end of a (FORWARD or BACKPROP) tick
// ------------------------------------------------------------------------
void w_advance_example (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "w_advance_example\n");
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

    // record the last example presented
    if (xcfg.training)
    {
      train_cnt = example_inx;
    }
    else
    {
      test_cnt = example_inx;
    }

    // reset example count for next epoch,
    example_cnt = 0;

    // and, if training, update weights and initialise weight changes
    //TODO: find a better place for this operation
    if (xcfg.training)
    {
      wb_update_func ();

      for (uint i = 0; i < wcfg.num_rows; i++)
      {
        for (uint j = 0; j < wcfg.num_cols; j++)
        {
          w_link_deltas[i][j] = 0;
        }
      }
    }
  }
  else
  {
    // fake network stop packet (expected only at end of epoch)
    //NOTE: safe to do it without disabling interrupts.
    net_stop_rdy = TRUE;
  }

  // start from first event for next example,
  evt = 0;
  num_events = ex[example_inx].num_events;

  // initialise unit outputs,
  for (uint i = 0; i < wcfg.num_rows; i++)
  {
    w_outputs[wf_procs][i] = wcfg.initOutput;
  }

  // access sync and net_stop flags with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // and check if can trigger next example computation
  if (net_stop_rdy)
  {
    // clear flag for next tick,
    net_stop_rdy = FALSE;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // and decide what to do
    if (net_stop)
    {
      // finish stage and report no error
      spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
    }
    else
    {
      // or trigger computation
      spin1_schedule_callback (wf_process, 0, 0, SPINN_WF_PROCESS_P);
    }
  }
  else
  {
    // flag as ready
    epoch_rdy = TRUE;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// switch from BACKPROP to FORWARD phase
// ------------------------------------------------------------------------
void w_switch_to_fw (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "w_switch_to_fw\n");
#endif

  // move to new FORWARD phase
  phase = SPINN_FORWARD;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// switch from FORWARD to BACKPROP phase
// ------------------------------------------------------------------------
void w_switch_to_bp (void)
{
#ifdef TRACE
  io_printf (IO_BUF, "w_switch_to_bp\n");
#endif

  // move to new BACKPROP phase,
  phase = SPINN_BACKPROP;

  // restore previous tick outputs,
  restore_outputs (tick - 1);

  // access queue and flag with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // check if need to schedule BACKPROP processing thread,
  if (!wb_active && (w_pkt_queue.tail != w_pkt_queue.head))
  {
    // flag as active,
    wb_active = TRUE;

    // restore interrupts after semaphore access,
    spin1_mode_restore (cpsr);

    // and schedule BACKPROP processing thread
    spin1_schedule_callback (w_processBKPQueue, 0, 0, SPINN_WB_PROCESS_P);
  }

  // and restore interrupts after queue and flag access
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------
