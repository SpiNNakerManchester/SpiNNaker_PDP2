// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "sdram.h"

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"
#include "activation.h"

// set of routines to be used by T core to process data

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint fwdKey;               // 32-bit packet ID for FORWARD phasees
extern uint bkpKey;               // 32-bit packet ID for BACKPROP phasees
extern uint stpKey;               // 32-bit packet ID for stop criterion

extern uint         epoch;        // current training iteration
extern uint         example;      // current example in epoch
extern uint         evt;          // current event in example
extern uint         num_events;   // number of events in current example
extern uint         event_idx;    // index into current event
extern proc_phase_t phase;        // FORWARD or BACKPROP
extern uint         num_ticks;    // number of ticks in current event
extern uint         max_ticks;    // maximum number of ticks in current event
extern uint         min_ticks;    // minimum number of ticks in current event
extern uint         tick;         // current tick in phase
extern uint         ev_tick;      // current tick in event
extern uchar        tick_stop;    // current tick stop decision
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// configuration structures (SDRAM)
// ------------------------------------------------------------------------
extern chip_struct_t        *ct; // chip-specific data
extern uint                 *cm; // simulation core map
extern uchar                *dt; // core-specific data
extern mc_table_entry_t     *rt; // multicast routing table data
extern weight_t             *wt; //# initial connection weights
extern mlp_set_t            *es; // example set data
extern mlp_example_t        *ex; // example data
extern mlp_event_t          *ev; // event data
extern activation_t         *it; // example inputs
extern activation_t         *tt; // example targets
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern global_conf_t  mlpc;       // network-wide configuration parameters
extern chip_struct_t  ccfg;       // chip configuration parameters
extern t_conf_t   tcfg;           // threshold core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// threshold core variables
// ------------------------------------------------------------------------
extern activation_t   * t_outputs;     // current tick unit outputs
extern net_t          * t_nets;        // nets received from sum cores
extern error_t        * t_errors[2];   // error banks: current and next tick
extern uint             t_it_idx;      // index into current inputs/targets
extern uint             t_tot_ticks;   // total ticks on current example
extern pkt_queue_t      t_net_pkt_q;   // queue to hold received nets
extern uchar            t_active;      // processing nets/errors from queue?
extern uchar            t_sync_done;   // have expected sync packets arrived?
extern activation_t   * t_last_integr_output;  //last integrator output value
extern llong_deriv_t  * t_last_integr_output_deriv; //last integrator output deriv value
extern activation_t   * t_out_hard_clamp_data; //values injected by hard clamps
extern activation_t   * t_out_weak_clamp_data; //values injected by weak clamps
extern uchar            t_hard_clamp_en;       //hard clamp output enabled
extern scoreboard_t     tf_arrived;    // keep track of expected nets
extern uint             tf_thrds_init; // sync. semaphore initial value
extern uint             tf_thrds_done; // sync. semaphore: proc & stop
extern uchar            tf_stop_prev;  // previous group stop criterion met?
extern uchar            tf_stop_crit;  // stop criterion met?
extern uchar            tf_stop_init;  // sync. semaphore: stop daisy chain
extern uchar            tf_stop_done;  // sync. semaphore: stop daisy chain
extern stop_crit_t      tf_stop_func;  // stop evaluation function
extern uint             tf_stop_key;   // stop criterion packet key
extern uint             tb_procs;      // pointer to processing errors
extern scoreboard_t     tb_arrived;    // keep track of expected errors
extern uint             tb_thrds_done; // sync. semaphore: proc & stop
extern int              t_max_output_unit; // unit with highest output
extern int              t_max_target_unit; // unit with highest target
extern activation_t     t_max_output;      // highest output value
extern activation_t     t_max_target;      // highest target value
// list of output pipeline procedures
extern out_proc_t const t_out_procs[SPINN_NUM_OUT_PROCS];
extern out_error_t const t_out_error[SPINN_NUM_ERROR_PROCS];
extern llong_deriv_t  * t_output_deriv;
extern llong_deriv_t  * t_output_deriv_history;
extern delta_t        * t_deltas;
extern activation_t   * t_target_history;
extern out_proc_back_t const t_out_back_procs[SPINN_NUM_OUT_PROCS];
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
#ifdef DEBUG
  extern uint pkt_sent;  // total packets sent
  extern uint sent_fwd;  // packets sent in FORWARD phase
  extern uint sent_bkp;  // packets sent in BACKPROP phase
  extern uint pkt_recv;  // total packets received
  extern uint spk_sent;  // sync packets sent
  extern uint spk_recv;  // sync packets received
  extern uint stp_sent;  // stop packets sent
  extern uint stp_recv;  // stop packets received
  extern uint wrng_phs;  // packets received in wrong phase
  extern uint wght_ups;  // number of weight updates done
  extern uint tot_tick;  // total number of ticks executed
#endif
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process FORWARD phase: compute outputs
// ------------------------------------------------------------------------
void tf_process (uint null0, uint null1)
{
  // process packet queue
  // access queue with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // process until queue empty
  while (t_net_pkt_q.head != t_net_pkt_q.tail)
  {
    // if not empty dequeue packet,
    uint key = t_net_pkt_q.queue[t_net_pkt_q.head].key;
    net_t net = (net_t) t_net_pkt_q.queue[t_net_pkt_q.head].payload;
    t_net_pkt_q.head = (t_net_pkt_q.head + 1) % SPINN_THLD_PQ_LEN;

    // restore interrupts after queue access,
    spin1_mode_restore (cpsr);

    // get net index: mask out block, phase and colour data,
    uint inx = (key & SPINN_NET_MASK);

    // store net for BACKPROP computation,
    t_nets[inx] = net;

    // compute unit output,
    //TODO: need to make sure this is the same as Lens
    compute_out (inx);

    activation_t activation = t_outputs[inx];

    // incorporate output index into packet key and send to next stage,
    while (!spin1_send_mc_packet ((fwdKey | inx),
                                   (uint) activation,
                                   WITH_PAYLOAD
                                 )
          );

    #ifdef DEBUG_VRB
      io_printf (IO_BUF, "o[%2d]:%11.7f (0x%08x)\n", inx,
                  SPINN_CONV_TO_PRINT(activation, SPINN_ACTIV_SHIFT),
                  activation
                );
    #endif
      
    #ifdef DEBUG
      pkt_sent++;
      sent_fwd++;
    #endif

    // evaluate stop criterion,
    if (tcfg.output_grp)
      tf_stop_func (inx);

    // mark net as arrived,
    #if SPINN_USE_COUNTER_SB == FALSE
      tf_arrived |= (1 << inx);
    #else
      tf_arrived++;
    #endif

    // and check if all nets arrived (i.e., all outputs done)
    if (tf_arrived == tcfg.f_all_arrived)
    {
      // if possible, FORWARD stop criterion
      if (tcfg.output_grp)
      {
        // check flags status in critical section
        uint cpsr = spin1_int_disable ();

        if (tf_stop_done == 0)
        {
          // initialize semaphore,
          tf_stop_done = tf_stop_init;

          // report stop criterion done,
          if (tcfg.is_last_output_group)
            tf_thrds_done -= 1;

          // restore interrupts after flag access,
          spin1_mode_restore (cpsr);

          // and send stop criterion packet
          //TODO: check if need to schedule or can simply call
          tf_send_stop (NULL, NULL);
        }
        else
        {
          // if not done report processing thread done,
          tf_stop_done -= 1;

          // and restore interrupts after flag access
          spin1_mode_restore (cpsr);
        }
      }

      // access synchronization semaphore with interrupts disabled
      uint cpsr = spin1_int_disable ();

      // and check if all threads done
      if (tf_thrds_done == 0)
      {
        // initialize semaphore,
        tf_thrds_done = tf_thrds_init;

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        //TODO: check if need to schedule or can simply call
        tf_advance_tick (NULL, NULL);
      }
      else
      {
        // if not done report processing thread done,
        tf_thrds_done -= 1;

        // and restore interrupts after flag access
        spin1_mode_restore (cpsr);
      }
    }

    // access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // when done flag going to sleep,
  t_active = FALSE;

  // and restore interrupts after queue access and leave
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP phase: compute error deltas
// ------------------------------------------------------------------------
void tb_process (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "tb_process\n");
  #endif

  // compute deltas based on pre-computed errors,
  //TODO: this needs checking!
  for (uint inx = 0; inx < tcfg.num_outputs; inx++)
  {
    delta_t delta;

    // update output derivatives for non-output groups
    //TODO: could avoid testing on every iteration
    if (!tcfg.output_grp)
    {
      // use received error computed in previous tick
      t_output_deriv[inx] = t_errors[tb_procs][inx];
    }

    // restore output derivatives for the tick being processed
    restore_output_deriv (inx);

    compute_out_back (inx);
    
    delta = t_deltas[inx];

    // send delta to input core for further processing
    while (!spin1_send_mc_packet ((bkpKey | inx), (uint) delta, WITH_PAYLOAD));

    #ifdef DEBUG
      pkt_sent++;
      sent_bkp++;
    #endif
    
    #ifdef DEBUG_VRB
      io_printf(IO_BUF, "d[%2d][%2d] = %10.7f (%08x)\n", tcfg.delta_blk, inx,
                 SPINN_CONV_TO_PRINT(delta, SPINN_DELTA_SHIFT),
                 delta
               );
    #endif
  }

  // access synchronization semaphore with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // and check if all threads done
  if (tb_thrds_done == 0)
  {
    // if done initialize synchronization semaphore,
    tb_thrds_done = 1;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // and advance tick
    //TODO: check if need to schedule or can simply call
    #ifdef TRACE_VRB
      io_printf (IO_BUF, "tbp calling tb_advance_tick\n");
    #endif

    tb_advance_tick (NULL, NULL);
  }
  else
  {
    // if not done report processing thread done,
    tb_thrds_done -= 1;

    // and restore interrupts after flag access
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void tf_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "tf_advance_tick\n");
  #endif
  
  // initialize scoreboard for next tick,
  tf_arrived = 0;

  // dump outputs to SDRAM for record keeping,
  #if SPINN_OUTPUT_HISTORY == TRUE
    //TODO: works only if examples have a single event
    //NOTE: could keep a pointer to current offset
    //NOTE: could use DMA
//###    spin1_memcpy(&oh[(((example * max_ticks) + tick) * mlpc.num_outs)
//###                     +
//###                     tcfg.output_offset
//###                    ],
//###                 t_outputs,
//###                 tcfg.num_outputs * sizeof(activation_t)
//###                );
  #endif

  // if requested report outputs to host,
  if (tcfg.write_out)
  {
    // is this the last report?
    if ((epoch    == (mlpc.num_epochs - 1))
         && (example == (mlpc.num_examples - 1))
         && (evt     == (num_events - 1))
         && (tick_stop)
       )
    {
      send_outputs_to_host (SPINN_HOST_FINAL, tick);
    }
    else
    {
      send_outputs_to_host (SPINN_HOST_NORMAL, tick);
    }
  }

  #ifdef DEBUG
    tot_tick++;
  #endif

  // and check if done with FORWARD phase
  if (tick_stop)
  {
    tf_advance_event ();
  }
  else
  {
    // if not done increment ticks
    tick++;
    ev_tick++;

    #ifdef TRACE
      io_printf (IO_BUF, "tf_tick: %d/%d\n", tick, tot_tick);
    #endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void tb_advance_tick (uint null0, uint null1)
{
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
    // initialize the tick count
    tick = SPINN_T_INIT_TICK;

    // initialize the event tick count
    ev_tick = SPINN_T_INIT_TICK;

    // switch to FORWARD phase,
    t_switch_to_fw ();

    // advance to next example,
    t_advance_example ();

    // and stop processing queue in this phase
    return;
  }
  else
  {
    // if not done decrement tick
    tick--;

    // and trigger computation
    spin1_schedule_callback (tb_process, NULL, NULL, SPINN_TB_PROCESS_P);

    #ifdef TRACE
      io_printf (IO_BUF, "tb_tick: %d/%d\n", tick, tot_tick);
    #endif
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
  
  // check if done with events,
  if (++evt >= num_events)
  {
    // check if in training mode
    if (mlpc.training)
    {
      // if training, save the number of ticks
      num_ticks = tick;
      // then do BACKPROP phase
      t_switch_to_bp ();

      // and stop processing queue in this phase
      return;
    }
    else
    {
      // if not training, initialize ticks for the next example
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
      t_it_idx += tcfg.num_outputs;

      // and update number of ticks for new event
      if (tcfg.is_last_output_group)
      {
        // maximum
        if (ev[event_idx + evt].max_time != SPINN_FP_NaN)
          max_ticks = (ev[event_idx + evt].max_time * mlpc.ticks_per_int)
            >> SPINN_FPREAL_SHIFT;
        else
          max_ticks = (es->max_time * mlpc.ticks_per_int) >> SPINN_FPREAL_SHIFT;
      
        // minimum
        if (ev[event_idx + evt].min_time != SPINN_FP_NaN)
          min_ticks = (ev[event_idx + evt].min_time * mlpc.ticks_per_int)
            >> SPINN_FPREAL_SHIFT;
        else
          min_ticks = (es->min_time * mlpc.ticks_per_int) >> SPINN_FPREAL_SHIFT;
      }
    }

    // increment example tick,
    tick++;

    // and initialize event tick
    ev_tick = SPINN_T_INIT_TICK;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: update the example at the end of a simulation tick
// ------------------------------------------------------------------------
void t_advance_example (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "t_advance_example\n");
  #endif
  
  // check if done with examples
  //TODO: alternative algorithms for chosing example order!
  if (++example >= mlpc.num_examples)
  {
    // check if done with epochs
    if (++epoch >= mlpc.num_epochs)
    {
      // done
      spin1_stop ();
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
  event_idx = ex[example].ev_idx;

  // if input or output group initialize new event input/target index
  if (tcfg.input_grp || tcfg.output_grp)
  {
    t_it_idx = ev[event_idx].it_idx * tcfg.num_outputs;
  }

  // update number of ticks for new event
  if (tcfg.is_last_output_group)
  {
    // maximum
    if (ev[event_idx + evt].max_time != SPINN_FP_NaN)
      max_ticks = (ev[event_idx + evt].max_time * mlpc.ticks_per_int)
        >> SPINN_FPREAL_SHIFT;
    else
      max_ticks = (es->max_time * mlpc.ticks_per_int) >> SPINN_FPREAL_SHIFT;

    // minimum
    if (ev[event_idx + evt].min_time != SPINN_FP_NaN)
      min_ticks = (ev[event_idx + evt].min_time * mlpc.ticks_per_int)
        >> SPINN_FPREAL_SHIFT;
    else
      min_ticks = (es->min_time * mlpc.ticks_per_int) >> SPINN_FPREAL_SHIFT;
  }

  // check if ready to send initial unit outputs,
  // access flags with interrupts disabled
  uint cpsr = spin1_int_disable ();

  if (t_sync_done)
  {
    // if ready clear synchronization flag,
    t_sync_done = FALSE;
  
    // restore interrupts,
    spin1_mode_restore (cpsr);

    // schedule sending of unit outputs,
    //TODO: check if need to schedule or can simply call
    spin1_schedule_callback (t_init_outputs, NULL, NULL, SPINN_T_INIT_OUT_P);

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
    // restore interrupts
    spin1_mode_restore (cpsr);
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
  
  // access queues with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // move to new FORWARD phase,
  phase = SPINN_FORWARD;

  // check if ready to start processing in FORWARD phase,
  //TODO: need to check this? -- see comms_t.c
  if (t_net_pkt_q.head != t_net_pkt_q.tail)
  {
    // if queue not empty schedule FORWARD processing
    spin1_schedule_callback (tf_process, NULL, NULL, SPINN_TF_PROCESS_P);
  }
  else
  {
    // if empty flag going inactive
    t_active = FALSE;
  }
  
  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: when the simulation is completed in the FORWARD phase,
// switch to the bacward phase if training is required 
// ------------------------------------------------------------------------
void t_switch_to_bp (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "t_switch_to_bp\n");
  #endif
  
  // access flags and queues with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // move to new BACKPROP phase,
  phase = SPINN_BACKPROP;

  // start processing in BACKPROP phase,
  //TODO: check!
  spin1_schedule_callback (tb_process, NULL, NULL, SPINN_TB_PROCESS_P);
  
  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// in the FORWARD phase the convergence critieron may require the simulation to
// stop before the maximum time is reached. This routine sends a broadcast
// message to communicate the final decision if the criterion has been reached
// across all the output groups to all the cores in teh simulation
// ------------------------------------------------------------------------
void tf_send_stop (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "tf_send_stop\n");
  #endif
  
  // "aggregate" criteria,
  tf_stop_crit = tf_stop_crit && tf_stop_prev;

  if (tcfg.is_last_output_group)
  {
    spin1_delay_us (2000); //##

    tf_stop_crit = (ev_tick == max_ticks)
                     || (tf_stop_crit && (ev_tick >= min_ticks));
    tick_stop = tf_stop_crit;
  }

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "M:%d t:%d sc:%x\n", max_ticks, ev_tick, tf_stop_crit);
  #endif

  // FORWARD aggregated criterion,
  while (!spin1_send_mc_packet ((tf_stop_key | tf_stop_crit),
                                 0,
                                 NO_PAYLOAD
                               )
        );

  #ifdef DEBUG
    stp_sent++;
  #endif

  // and initialize criterion for next tick
  tf_stop_crit = TRUE;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// this routine initializes the output values of the units. There is a conflict
// in the initialization routine between lens 2.63 and lens 2.64.
// The current version implements the routine as expressed by lens 2.63, with
// comments on the line to change to apply lens version 2.64
// ------------------------------------------------------------------------
void t_init_outputs (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "t_init_outputs\n");
  #endif
  
  // initialize every unit output and send for processing
  for (uint i = 0; i < tcfg.num_outputs; i++)
  {
    // setup the initial output value.
    // Lens has two ways of initialise the output value, as defined in Lens 2.63
    // and Lens 2.64, and the two ways are not compatible

/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
//NOTE: The following code follows modification MOD009 of Lens version 2.64:
/******************************************************************************/
/*  MOD009: (09/10/05) (i) Bug fix for continuous network(courtesy Dave Plaut)*/
/******************************************************************************/
// it has been commented out as per e-mail interchange with Stephen Welbourne on 20th Sep 2013
// the lens code as per MOD009 is the following, from: void resetOutputs(Group G) in act.c in Lens
//
//  if (!G->inputProcs || G->outputType & HARD_CLAMP) {
//    FOR_EACH_UNIT2(G, U->output = O[u] = (isNaN(U->externalInput) ? initOutput : U->externalInput));
//  } else {
//    FOR_EACH_UNIT2(G, U->output = O[u] = initOutput);
//  }
//
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/

    // use initial values,
    // TODO: need to verify initInput with Lens    
    // NOTE: The following code follows the output of Lens 2.63:
    // initialise the output value of the units
    t_outputs[i] = tcfg.initOutput;
    
    // if the output integrator is used
    // reset the array of the last values
    if (tcfg.out_integr_en) {
      t_last_integr_output[i] = tcfg.initOutput;
      t_last_integr_output_deriv[i] = 0;
    }

    // and send unit output to weight cores
    while (!spin1_send_mc_packet ((fwdKey | i),
                                   (uint) t_outputs[i],
                                   WITH_PAYLOAD
                                 )
          );

    #ifdef DEBUG
      pkt_sent++;
      sent_fwd++;
    #endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// This routine calls in the appropriate order all the elements of the output
// pipeline, as expressed in the array passed by splens tcfg.procs_list[i].
// the routines to be called are listed in the array t_out_procs[]
// The SPINN_STORE_OUTPUT and the SPINN_STORE_TARGET flags are currently set
// at compile time, but they should be passed through the tcfg structure.
// This has also an implication in the initialization routine, as the memory
// where to store the history needs to be allocated in SDRAM.
// Finally also the return values of the error function (output_deriv) need to
// be stored to be used in the BACKPROP phase
// The output pipeline is computer for each single unit (inx) before passing to
// the next unit
// ------------------------------------------------------------------------
void compute_out (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "compute_out\n");
  #endif

  #ifdef DEBUG_VRB
    char* group;
    group = (tcfg.input_grp) ? "Input" : ((tcfg.output_grp) ? "Output" : ((tcfg.num_outputs == 1) ? "Bias" : "Hidden"));
    io_printf (IO_BUF, "compute_out - Group: %s - Example: %d - Tick: %d, Unit: %d\n", group, example, tick, inx);
  #endif

  // initialize the array element where to store the output value for the 
  t_outputs[inx] = 0;
  
  // compute all the elements of the output pipeline
  // from the observations in lens, the logistic is always the first element of
  // the output pipeline, which uses the value received through the multicast
  // packet. If no logistic function is used, the t_outputs starts with a 0
  // value, as initialized earlier
  for (uint i = 0; i < tcfg.num_out_procs; i++)
  {
    t_out_procs[tcfg.procs_list[i]] (inx);
  }

  // if the network is set for training, then compute the output derivative
  // using the appropriate function as set by splens
  if (mlpc.training && tcfg.output_grp)
  {
    #ifdef TRACE_VRB
      io_printf (IO_BUF, "compute output deriv\n");
    #endif

    // if the error function to be called is not null compute the output derivative
    if (t_out_error[tcfg.error_function] != NULL)
    {
      t_out_error[tcfg.error_function] (inx);
    }
  }

  // check if in training mode, and if so, store outputs, targets, and output derivatives
  // TODO: for non-continuous networks, this needs to check the requirement to have these 
  // histories saved, which needs to come from splens. For continuous networks, these histories
  // are always required. 
  if (mlpc.training)
  {
    store_targets (inx);
    store_output_deriv (inx);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores the targets for the current tick
// ------------------------------------------------------------------------
void store_targets (uint inx)
{
  #ifdef TRACE
    io_printf (IO_BUF, "store_targets\n");
  #endif

  activation_t * src_ptr = &tt[t_it_idx + inx];
  activation_t * dst_ptr = t_target_history + (((tick-1) * tcfg.num_outputs) + inx);

  spin1_memcpy(dst_ptr, src_ptr, sizeof(activation_t));
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores the output derivatives for the current tick
// ------------------------------------------------------------------------
void store_output_deriv (uint inx)
{
  #ifdef TRACE
    io_printf (IO_BUF, "store_output_deriv\n");
  #endif

  llong_deriv_t * src_ptr = t_output_deriv + inx;
  llong_deriv_t * dst_ptr = t_output_deriv_history + (((tick-1) * tcfg.num_outputs) + inx);
    
  spin1_memcpy(dst_ptr, src_ptr, sizeof(llong_deriv_t));
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores the output derivative for the specified unit and the 
// current value of the global variable tick.
// ------------------------------------------------------------------------
void restore_output_deriv (uint inx)
{
  #ifdef TRACE
    io_printf (IO_BUF, "restore_output_deriv\n");
  #endif
  t_output_deriv[inx] = t_output_deriv_history[(((tick-1) * tcfg.num_outputs) + inx)];
}
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// compute the logistic function starting from the value received through the 
// multicast packet.
// ------------------------------------------------------------------------
void out_logistic (uint inx)
{
  #ifdef TRACE_VRB
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
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_integr\n");
  #endif
  
  // representation 49.15, with the topmost 48 bits set to 0
  llong_activ_t last_output = t_last_integr_output[inx]; 
  // representation 49.15, with the topmost 48 bits set to 0
  llong_activ_t new_output = t_outputs[inx];
  // representation 48.16, with the topmost 32 bits set to 0
  lfpreal dt = tcfg.out_integr_dt;
  
  // compute the output integration following Lens code
  // representation: 
  // 49.15 + ((48.16 * (49.15 - 49.15)) >> 16) = 
  // = 49.15 + (48.16 * 49.15) >> 16 = 49.15 + 49.15 =
  // = 49.15
  // The values are unlikely to overflow, as the variables are extended to
  // 64 bits for this purpose and the 32 or 48 topmost bits are set to 0 by the
  // extension
  llong_activ_t output = last_output + ((dt * (new_output - last_output)) >> 16);

  // saturate the value computed and assign it to the output variable
  if (output > (llong_activ_t) SPINN_ACTIV_MAX)
    // positive saturation
    t_outputs[inx] = (activation_t) SPINN_ACTIV_MAX;  
  else if (output < (llong_activ_t) SPINN_ACTIV_MIN_NEG)
    // negative saturation
    t_outputs[inx] = (activation_t) SPINN_ACTIV_MIN_NEG;
  else
    // representation in 49.15 within the range (-1; 1) can be reduced to 1.15
    t_outputs[inx] = (activation_t) output;
  
  // store the integrator state for the next iteration
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
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_hard_clamp\n");
  #endif

  // compute only if input is not NaN
  if (it[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    // assign the value coming from the event
    t_outputs[inx] = it[t_it_idx + inx];
  }

  // TODO: if training, store the injected value in SDRAM. This memory area needs
  // to be allocated during initialization
/*
  if (mlpc.training)
  {
    activation_t * tmp = t_out_hard_clamp_data + tick * tcfg.num_outputs;
    tmp[inx] = t_outputs[inx];
  }
*/
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the bias clamp. This clamp is used only by the bias group, and sets
// the output value to a constant maximum output value (which is close to 1
// on spinnaker, and 1 in lens
// ------------------------------------------------------------------------
void out_bias (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_bias\n");
  #endif

  // set output value to SPINN_ACTIV_MAX (close to 1)
  t_outputs[inx] = (activation_t) SPINN_ACTIV_MAX;
}
// ------------------------------------------------------------------------


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
 * Lens computation of the weak clamp operator
 * 
 * static void weakClampOutput(Group G, GroupProc P) {
 *   real *originalOutput = P->unitHistoryData[HISTORY_INDEX(Net->currentTick)],
 *     strength = chooseValue(G->clampStrength, Net->clampStrength);
 *   FOR_EACH_UNIT2(G, {
 *     originalOutput[u] = U->output;
 *     if (!isNaN(U->externalInput))
 *       U->output += strength * (U->externalInput - U->output);
 *   });
 * }
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


// ------------------------------------------------------------------------
// compute the weak clamp, as defined by lens, and store the injected value, if
// the network needs training
// ------------------------------------------------------------------------
void out_weak_clamp (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_weak_clamp\n");
  #endif
  
  // TODO: if training, store the injected value in SDRAM. This memory area needs
  // to be allocated during initialization
/*
  if (mlpc.training)
  {
    //store previous value of t_output for BACKPROP computation
    activation_t * tmp = t_out_weak_clamp_data + tick * tcfg.num_outputs;
    tmp[inx] = t_outputs[inx];
  }
*/

  llong_activ_t external_input = it[t_it_idx + inx];        // 49.15 repr.

  // compute only if input is not NaN
  if (external_input != (llong_activ_t) SPINN_ACTIV_NaN)
  {
    lfpreal weak_clamp_strength = tcfg.weak_clamp_strength; // 48.16 repr.
    llong_activ_t output_value = t_outputs[inx];            // 49.15 repr.
    
    // computation of the weak clamp output following Lens implementation 
    // representation: 49.15 + (48.16 * (49.15 - 49.15) >> 16) = 49.15
    llong_activ_t output = output_value
                             + ((weak_clamp_strength
                                 * (external_input - output_value))
                                   >> SPINN_FPREAL_SHIFT
                               );
    
    // saturate output and cast into 1.15 representation
    if (output > (llong_activ_t) SPINN_ACTIV_MAX)
      t_outputs[inx] = (activation_t) SPINN_ACTIV_MAX;  
    else if (output < (llong_activ_t) SPINN_ACTIV_MIN_NEG)
      t_outputs[inx] = (activation_t) SPINN_ACTIV_MIN_NEG;
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
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "compute_out_back\n");
  #endif

  #ifdef DEBUG_VRB
    char* group;
    group = (tcfg.input_grp) ? "Input" : ((tcfg.output_grp) ? "Output" : ((tcfg.num_outputs == 1) ? "Bias" : "Hidden"));
    io_printf (IO_BUF, "compute_out_back - Group: %s - Example: %d - Tick: %d - Unit: %d\n", group, example, tick, inx);
  #endif
  
  int i;
  
  // if the output pipeline includes one or more elements, compute them in the
  // reverse order
  if (tcfg.num_out_procs >= 1)
  {
    for (i = tcfg.num_out_procs-1; i >= 0; i--)
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
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_logistic_back\n");
  #endif

  //NODE: may need to use a longer type and saturate!
  // compute error delta,
  // keep the correct implicit decimal point position
  // s16.15 = (s48.15 * s0.15) >> 15
  t_deltas[inx] = (delta_t) (((long_delta_t) t_output_deriv[inx]
                    * (long_delta_t) sigmoid_prime (t_nets[inx]))
                    >> (SPINN_DERIV_SHIFT + SPINN_ACTIV_SHIFT 
                    - SPINN_DELTA_SHIFT));
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the output integration operation for the backprop
// ------------------------------------------------------------------------
void out_integr_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_integr_back\n");
  #endif

  llong_deriv_t last_output_deriv = t_last_integr_output_deriv[inx];

  lfpreal dt = tcfg.out_integr_dt;

  // s48.15 = (s47.16 * s48.15) >> 16
  llong_deriv_t d = (dt * last_output_deriv) >> 16;
  last_output_deriv += t_output_deriv[inx] - d;
  t_output_deriv[inx] = d;
  
  // store the integrator state for the next iteration
  t_last_integr_output_deriv[inx] = last_output_deriv;
}
// ------------------------------------------------------------------------


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
static void hardClampOutputBack(Group G, GroupProc P) {
  printf ("hardClampOutputBack\n");
  int tick = HISTORY_INDEX(Net->currentTick);
  real *externalInputHistory = P->unitHistoryData[tick];
  FOR_EACH_UNIT2(G, {
    if (!isNaN(externalInputHistory[u]))
      U->inputDeriv = 0.0;
  });
}
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


// ------------------------------------------------------------------------
// TODO: BACKPROP phase for the hard clamp - this is a stub
// ------------------------------------------------------------------------
void out_hard_clamp_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_hard_clamp_back\n");
  #endif

/*
  activation_t * tmp = t_out_hard_clamp_data + tick * tcfg.num_outputs;
  
  if (tmp[inx] != SPINN_ACTIV_NaN)
    t_output_deriv[inx] = 0;
*/
}
// ------------------------------------------------------------------------


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
static void weakClampOutputBack(Group G, GroupProc P) {
  real *originalOutput = P->unitHistoryData[HISTORY_INDEX(Net->currentTick)],
    scale = 1.0 - chooseValue(G->clampStrength, Net->clampStrength);
  FOR_EACH_UNIT2(G, {
    if (U->output != originalOutput[u]) {
      U->output = originalOutput[u];
      U->outputDeriv *= scale;
    }
  });
}
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


// ------------------------------------------------------------------------
// TODO: BACKPROP phase for the weak clamp - for the moment is a stub
// ------------------------------------------------------------------------
void out_weak_clamp_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_weak_clamp_back\n");
  #endif
}
// ------------------------------------------------------------------------


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
static void biasClampOutputBack(Group G, GroupProc P) {
  printf ("biasClampOutputBack\n");
  FOR_EACH_UNIT(G, U->inputDeriv = 0.0);
}
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


// ------------------------------------------------------------------------
// TODO: BACKPROP phase for the bias clamp
// ------------------------------------------------------------------------
void out_bias_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "out_bias_back\n");
  #endif

  t_output_deriv[inx] = 0;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// initialization code for the output integrator: allocate the memory to save
// the state of the integrator and initialize the state to 0
// FIXME: to be checked - the initialization may be superfluous as the value is
// set to initoutput in the following initialization steps
// ------------------------------------------------------------------------
int init_out_integr ()
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "init_out_integr\n");
  #endif

  int i;

  // allocate memory for integrator state
  if ((t_last_integr_output = ((activation_t *)
       spin1_malloc (tcfg.num_outputs * sizeof(activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((t_last_integr_output_deriv = ((llong_deriv_t *)
       spin1_malloc (tcfg.num_outputs * sizeof(llong_deriv_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  for (i = 0; i<tcfg.num_outputs; i++)
    t_last_integr_output[i] = 0;
    t_last_integr_output_deriv[i] = 0;

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// initialization of the hard clamp includes SDRAM memory allocation to store
// information related to the values injected. This function is currently a stub
// ------------------------------------------------------------------------
int init_out_hard_clamp ()
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "init_out_hard_clamp\n");
  #endif

/*
  if (mlpc.training)
  {
    // allocate memory for outputs
    if ((t_out_hard_clamp_data = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_outputs * mlpc.global_max_ticks * sizeof(activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
       )
    {
      return (SPINN_MEM_UNAVAIL);
    }
  }
  
  io_printf (IO_BUF, "hc store addr %08x\n", (int) t_out_hard_clamp_data);
*/

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// initialization of the hard clamp includes SDRAM memory allocation to store
// information related to the values injected. This function is currently a stub
// ------------------------------------------------------------------------
int init_out_weak_clamp ()
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "init_out_weak_clamp\n");
  #endif

/*
  if (mlpc.training)
  {
    // allocate memory for outputs
    if ((t_out_weak_clamp_data = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_outputs * mlpc.global_max_ticks * sizeof(activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
       )
    {
      return (SPINN_MEM_UNAVAIL);
    }
  }
*/

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// evaluation of the standard convergence criteria.
// for each unit in the output group check if the output value is close
// to the target value, with the "group_criterion" variable defining the
// acceptance margin
// ------------------------------------------------------------------------
void std_stop_crit (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "std_stop_crit\n");
  #endif

  // evaluate only if target is not NaN
  if (tt[t_it_idx + inx] != SPINN_ACTIV_NaN)
  {
    error_t error = ABS (t_outputs[inx] - tt[t_it_idx + inx]);
    
    // Correction to fixed point arithmetic: tcfg.group_criterion is assumed
    // to be in a format with 15 decimal bits, but appears to have 16, making
    // it twice the size it should be.  Therefore shift one bit to the right.
    tf_stop_crit = tf_stop_crit && (error < (tcfg.group_criterion >> 1));
  }
}
// ------------------------------------------------------------------------


/*
// ------------------------------------------------------------------------
// The following routine has only been used for debugging purposes
// and the only scope for it is to run the simulation always for the maximum
// number of ticks
// ------------------------------------------------------------------------
void max_stop_crit (uint inx)  //## DEBUGGING
{
  tf_stop_crit = FALSE;
}
*/


// ------------------------------------------------------------------------
// evaluation of the "max" convergence criteria.
// for each unit in the output group check if both the output and target values
// are the maximum in the group and, in this case, if their difference is less
// or equal than the tcfg.group_criterion value.
// TODO: this routine needs to be modified to adapt to the case in which the
// output group is split across multiple cores, as this i sa global convergence
// rule, rather than an individual one, as the standard convergence criterion
// ------------------------------------------------------------------------
void max_stop_crit (uint inx)
{
  #ifdef TRACE_VRB
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

    error_t error = ABS (t_max_output - t_max_target);

    // Correction to fixed point arithmetic: tcfg.group_criterion is assumed
    // to be in a format with 15 decimal bits, but appears to have 16, making
    // it twice the size it should be.  Therefore shift one bit to the right.
    if ((t_max_output_unit == -1)
	 || ((t_max_output_unit == t_max_target_unit)
	     && (error < (tcfg.group_criterion >> 1))
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
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "error_squared\n");
  #endif

  if (tt[t_it_idx + inx] != SPINN_ACTIV_NaN)
    t_output_deriv[inx] = ((llong_deriv_t) t_outputs[inx] - (llong_deriv_t) tt[t_it_idx + inx]) << 1;
  else
    t_output_deriv[inx] = 0;
}
// ------------------------------------------------------------------------


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
#define SMALL_VAL ((double) 1e-8)
#define LARGE_VAL ((double) (SPINN_LONG_ACTIV_MAX >> SPINN_ACTIV_SHIFT)+1)

#define SIMPLE_CED(y, d)\
     ((((real) (y)*(1.0-(y))) <= SMALL_VAL) ?\
     (((real) (y)-(d)) * LARGE_VAL) :\
       (((real) (y)-(d))/((real) (y)*(1.0-(y)))))
       
#define CED_ZERO_TARGET(y)\
     (((real) (1.0-(y)) <= SMALL_VAL) ? LARGE_VAL :\
     ((real) 1.0/(1.0-(y))))
     
#define CED_ONE_TARGET(y)\
     (((real) (y) <= SMALL_VAL) ? -LARGE_VAL :\
     ((real) -1.0/(y)))
     
#define CROSS_ENTROPY_DERIV(y, d)\
     (((real) (d) == 0.0) ? CED_ZERO_TARGET(y) :\
       (((real) (d) == 1.0) ? CED_ONE_TARGET(y) :\
     SIMPLE_CED(y, d)))
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


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
  #ifdef TRACE_VRB
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
      if ((long_activ_t) SPINN_LONG_ACTIV_ONE - t_outputs[inx] <= (long_activ_t) SPINN_SMALL_VAL)
      {
        t_output_deriv[inx] = (llong_deriv_t) SPINN_LONG_DERIV_MAX;
      }
      // otherwise compute 1 / (1 - output)
      else
      {
        long_deriv_t numerator = (long_deriv_t) SPINN_LONG_DERIV_ONE; //representation: 17.15
        long_deriv_t denominator = (long_deriv_t) SPINN_LONG_DERIV_ONE - t_outputs[inx]; //representation: 17.15
  
        // the left shift needs to be done before the division, as the
        // precision reduces with the division
        // representation: 49.15
        t_output_deriv[inx] = (((llong_deriv_t) numerator << SPINN_DERIV_SHIFT) / (llong_deriv_t) denominator);
      }
    }
    // if the target is close to 1, then the cross entropy function simplifies:
    // -1 / output
    else if (tt[t_it_idx + inx] == (long_activ_t) SPINN_LONG_ACTIV_ONE)
    {
      // if the output value is close to 0, then the cross entropy function
      // shows a discontinuity, and the output value is set to the minimum
      // negative value that can be represented
      if (t_outputs[inx] <= (long_activ_t) SPINN_SMALL_VAL)
      {
        t_output_deriv[inx] = (llong_deriv_t) SPINN_LONG_DERIV_MIN_NEG;
      }
      // otherwise compute -1 / output
      else
      {
        long_deriv_t numerator = (long_deriv_t) SPINN_LONG_DERIV_NEG_ONE; //representation: 17.15
        long_deriv_t denominator = t_outputs[inx]; //representation: 17.15

        // the left shift needs to be done before the division, as the
        // precision reduces with the division
        // representation: 49.15
        t_output_deriv[inx] = (((llong_deriv_t) numerator << SPINN_DERIV_SHIFT) / (llong_deriv_t) denominator);
      }
    }
    // otherwise compute the standard function
    else
    {
      // if (output * (1-output)) is close to 0, the function presents a
      // discontinuity, and the result is computed as
      // MAX_VALUE * (output - target)
      // where the MAX value is the maximum representable value
      if (( ((llong_activ_t) t_outputs[inx] * (llong_activ_t) ((long_activ_t) SPINN_LONG_ACTIV_ONE - t_outputs[inx])) << (llong_activ_t) SPINN_ACTIV_SHIFT) <= (long_activ_t) SPINN_SMALL_VAL)
      {
        t_output_deriv[inx] = ((((llong_deriv_t) SPINN_LONG_DERIV_MAX) * (llong_deriv_t)(t_outputs[inx] - tt[t_it_idx + inx])) >> SPINN_DERIV_SHIFT);
      }
      // otherwise compute the standard formula
      // (output - target) / (output * (1 - output))
      else
      {
        long_deriv_t numerator = ((long_deriv_t) t_outputs[inx] - (long_deriv_t) tt[t_it_idx + inx]); //representation: 17.15
        long_deriv_t one = (long_deriv_t) SPINN_LONG_DERIV_ONE; //representation: 17.15
        llong_deriv_t denominator = ((llong_deriv_t) t_outputs[inx] * (llong_deriv_t) (one - t_outputs[inx])) >> SPINN_DERIV_SHIFT; //representation: 49.15
        
        // representation: 49.15
        t_output_deriv[inx] = (((llong_deriv_t) numerator << SPINN_DERIV_SHIFT) / (llong_deriv_t) denominator);
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
