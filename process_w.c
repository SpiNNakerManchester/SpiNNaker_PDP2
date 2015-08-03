// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "sdram.h"

#include "init_w.h"
#include "comms_w.h"
#include "process_w.h"
#include "activation.h"

// set of routines to be used by W core to process data

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint fwdKey;               // 32-bit packet ID for FORWARD phase
extern uint bkpKey;               // 32-bit packet ID for BACKPROP phase
extern uint stpKey;               // 32-bit packet ID for stop criterion

extern uint         epoch;        // current training iteration
extern uint         example;      // current example in epoch
extern uint         evt;          // current event in example
extern uint         num_events;   // number of events in current example
extern proc_phase_t phase;        // FORWARD or BACKPROP
extern uint         num_ticks;    // number of ticks in current event
extern uint         tick;         // current tick in phase
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
extern w_conf_t       wcfg;       // weight core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// weight core variables
// ------------------------------------------------------------------------
extern weight_t     * * w_weights;     // connection weights block
extern wchange_t    * * w_wchanges;    // accumulated weight changes
extern activation_t   * w_outputs[2];  // unit outputs for b-d-p
extern delta_t        * w_deltas;      // error deltas for b-d-p
extern error_t        * w_errors;      // computed errors next tick
extern pkt_queue_t      w_delta_pkt_q; // queue to hold received deltas
extern uint             wf_procs;      // pointer to processing unit outputs
extern uint             wf_thrds_done; // sync. semaphore: comms, proc & stop
extern uint             wf_sync_key;   // FORWARD processing can start
extern uchar            wb_active;     // processing deltas from queue?
extern scoreboard_t     wb_arrived;    // keeps track of received deltas
extern uint             wb_sync_key;   // BACKPROP processing can start
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
#ifdef DEBUG
  extern uint pkt_sent;  // total packets sent
  extern uint sent_fwd;  // packets sent in FORWARD phase
  extern uint sent_bkp;  // packets sent in BACKPROP phase
  extern uint pkt_recv;  // total packets received
  extern uint recv_fwd;  // packets received in FORWARD phase
  extern uint recv_bkp;  // packets received in BACKPROP phase
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
// process FORWARD phase: compute partial dot products (output * weight)
// ------------------------------------------------------------------------
void wf_process (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "wf_process\n");
  #endif
  
  // compute all net block dot-products and send them for accumulation,
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
    //net_t is a 32 bit value with 27 decimal bits
    net_t net_part = 0;

    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      //w_outputs is a two-dimensional array of activation_t types
      //activation_t is a 16 bit value with 15 decimal bits
      //w_weights is a two-dimensional array of weight_t types
      //weight_t is a 16 bit value with 12 decimal bits and one sign bit
      //net_t is a 32 bit value with 27 decimal bits
      //1,15 * s3,12 = s4,27
      //NOTE: may need to use long_nets for the dot-products and saturate!
      net_part += ((net_t) w_outputs[wf_procs][i] * (net_t) w_weights[i][j]);
    }

//    if (epoch == 0 && example == 0 && tick == 1)
//    {
//      io_printf (IO_BUF,
//                  "w_nets[%d] for update %d example %d tick %d unit %d: %r\n",
//                  j, epoch, example, tick, j, (net_part >> 12)
//                );
//    }

    // incorporate net index to the packet key and send
    while (!spin1_send_mc_packet ((fwdKey | j), (uint) net_part, WITH_PAYLOAD));
    
    #ifdef DEBUG
      pkt_sent++;
      sent_fwd++;
    #endif
  }

  // access synchronization semaphore with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // and check if all threads done
  if (wf_thrds_done == 0)
  {
    // if done initialize synchronization semaphore,
    wf_thrds_done = 2;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // and advance tick
    //TODO: check if need to schedule or can simply call
    #ifdef TRACE_VRB
      io_printf (IO_BUF, "wfp calling wf_advance_tick\n");
    #endif

    wf_advance_tick (NULL, NULL);
  }
  else
  {
    // if not done report processing thread done,
    wf_thrds_done -= 1;

    // and restore interrupts after flag access
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP phase: compute partial products (weight * delta)
// ------------------------------------------------------------------------
void wb_process (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "wb_process\n");
  #endif
  
  #ifdef PROFILE
    io_printf (IO_STD, "tin:  %u\n", tc[T2_COUNT]);
  #endif

  // process delta packet queue
  // access queue with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // process until queue empty
  while (w_delta_pkt_q.head != w_delta_pkt_q.tail)
  {
    // if not empty dequeue packet,
    uint inx = w_delta_pkt_q.queue[w_delta_pkt_q.head].key;
    uint delta = (delta_t) w_delta_pkt_q.queue[w_delta_pkt_q.head].payload;
    w_delta_pkt_q.head = (w_delta_pkt_q.head + 1) % SPINN_WEIGHT_PQ_LEN;

    // restore interrupts after queue access,
    spin1_mode_restore (cpsr);

    // get delta index: mask out phase, core and block data,
    inx &= SPINN_DELTA_MASK;

    // store received error delta,
    w_deltas[inx] = delta;

    // update scoreboard,
    #if SPINN_USE_COUNTER_SB == FALSE
      wb_arrived |= (1 << inx);
    #else
      wb_arrived++;
    #endif

    // partially compute error dot products,
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      //TODO: may need to use long_error for the dot_products and saturate!
      /*err_part += ((long_error_t) w_weights[i][j]
                    * (long_error_t) w_deltas[wb_procs][j]
                  ) >> (LONG_ERR_SHIFT - ERROR_SHIFT);*/
      w_errors[inx] += (error_t) w_weights[i][inx] * (error_t) delta;
    
      //TODO: need to compute "link derivative" here (see w_weight_deltas)

      // check if done with all deltas
      if (wb_arrived == wcfg.b_all_arrived)
      {
        // send computed error dot product,
        while (!spin1_send_mc_packet ((bkpKey | i),
                (uint) w_errors[inx], WITH_PAYLOAD)
              );

        #ifdef DEBUG
          pkt_sent++;
          sent_bkp++;
        #endif

        // and initialize error for next tick
        w_errors[inx] = 0;
      }
    }

    // if done with all deltas advance tick
    if (wb_arrived == wcfg.b_all_arrived)
    {
      // initialize arrival scoreboard for next tick,
      wb_arrived = 0;

      #ifdef TRACE_VRB
        io_printf (IO_BUF, "wbp calling wb_advance_tick\n");
      #endif

      //TODO: check if need to schedule or can simply call
      wb_advance_tick (NULL, NULL);
    }

    // access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // when done, flag that going to sleep,
  wb_active = FALSE;

  // restore interrupts and leave
  spin1_mode_restore (cpsr);

  #ifdef PROFILE
    io_printf (IO_STD, "tout: %u\n", tc[T2_COUNT]);
  #endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// perform a weight update
// a weight of 0 means that there is no connection between the two units.
// the zero value is represented by the lowest possible (positive or negative)
// weight. A weight value is a 4.12 variable in fixed point
// ------------------------------------------------------------------------
void w_update_weights (void)
{
  #ifdef DEBUG
    wght_ups++;
  #endif

  #ifdef TRACE
    io_printf (IO_BUF, "w_update_weights\n");
  #endif
  
  // update weights
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      #ifdef DEBUG_VRB
        weight_t old_weight = w_weights[i][j];
      #endif

      // do not update weights that are 0 -- indicates no connection!
      if (w_weights[i][j] != 0)
      {
        // compute new weight
        weight_t temp = w_weights[i][j] + w_wchanges[i][j];

        // saturate new weight,
        if (temp >= SPINN_WEIGHT_MAX)
        {
          w_weights[i][j] = SPINN_WEIGHT_MAX;
        }
        else if (temp <= SPINN_WEIGHT_MIN)
        {
          w_weights[i][j] = SPINN_WEIGHT_MIN;
        }
        // and avoid (new weight == 0) -- indicates no connection!
        else if (temp == 0)
        {
          if (w_weights[i][j] > 0)
          {
            w_weights[i][j] = SPINN_WEIGHT_POS_DELTA;
          }
          else
          {
            w_weights[i][j] = SPINN_WEIGHT_NEG_DELTA;
          }
        }
        else
        {
          w_weights[i][j] = temp;
        }
      }

      #ifdef DEBUG_VRB
        uint roff = wcfg.blk_row * wcfg.num_rows;
        uint coff = wcfg.blk_col * wcfg.num_cols;

        io_printf (IO_BUF,
                    "[%2d][%2d] wo = %10.7f (0x%08x) wn = %10.7f (0x%08x)\n",
                    roff + i, coff + j,
                    SPINN_CONV_TO_PRINT(old_weight, SPINN_WEIGHT_SHIFT),
                    old_weight,
                    SPINN_CONV_TO_PRINT(w_weights[i][j], SPINN_WEIGHT_SHIFT),
                    w_weights[i][j]
                  );
      #endif
    }
  }

  #if SPINN_WEIGHT_HISTORY == TRUE
    // dump weights to SDRAM for record keeping
    //TODO: broken -- needs fixing!
    //TODO: works only if examples have a single event
    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      //NOTE: could use DMA
//##      spin1_memcpy(&wh[(((example + 1) * mlpc.num_outs
//##                        + wcfg.blk_row * wcfg.num_rows + i) * mlpc.num_outs)
//##                        + (wcfg.blk_col * wcfg.num_cols)],
//##                   w_weights[i],
//##                   wcfg.num_cols * sizeof(weight_t)
//##                  );
    }
  #endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute weight changes and store them in the w_wchanges matrix
// to be applied afterwards, through the weight update routine
// ------------------------------------------------------------------------
void w_weight_deltas (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "w_weight_deltas\n");
  #endif
  
  // compute weight changes
  for (uint j = 0; j < wcfg.num_cols; j++)
  {
    wchange_t temp = wcfg.learningRate * w_deltas[j];
    
    #ifdef DEBUG_VRB
      io_printf (IO_BUF, "t = %10.7f (0x%08x)\n",
                  SPINN_LCONV_TO_PRINT(temp,
                                        (SPINN_ACTIV_SHIFT + SPINN_DELTA_SHIFT)
                                      ),
                  temp
                );
    #endif

    for (uint i = 0; i < wcfg.num_rows; i++)
    {
      // never update weights that are 0! -- indicates no connection
      if (w_weights[i][j] != 0)
      {
        // keep the correct implicit decimal point position
        //NOTE: use the correct outputs from FORWARD phase
        w_wchanges[i][j] += (((long_error_t) temp *
                              (long_error_t) w_outputs[wf_procs][i]  //#
			     ) >> (2 * SPINN_ACTIV_SHIFT)
                            );
      }

      #ifdef DEBUG_VRB
        uint roff = wcfg.blk_row * wcfg.num_rows;
        uint coff = wcfg.blk_col * wcfg.num_cols;
        io_printf (IO_BUF,
                    "[%2d][%2d] w = %10.7f c = %10.7f d = %10.7f\n",
                    roff + i, coff + j,
                    SPINN_CONV_TO_PRINT(w_weights[i][j],  SPINN_WEIGHT_SHIFT),
                    SPINN_CONV_TO_PRINT(w_wchanges[i][j], SPINN_WEIGHT_SHIFT),
                    SPINN_CONV_TO_PRINT(w_deltas[wb_procs][j],
                                         SPINN_DELTA_SHIFT
                                       )
                  );
      #endif
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void wf_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "wf_advance_tick\n");
  #endif

  // update pointer to processing unit outputs,
  wf_procs = 1 - wf_procs;

  // check if end of example's FORWARD phase
  if (tick_stop)
  {
    wf_advance_event ();
  }
  else
  {
    // if not increment tick,
    tick++;

    #ifdef DEBUG
      tot_tick++;
    #endif

    // change packet key colour,
    fwdKey ^= SPINN_COLOUR_KEY;

    // and trigger computation
    spin1_schedule_callback (wf_process, NULL, NULL, SPINN_WF_PROCESS_P);

    #ifdef TRACE
      io_printf (IO_BUF, "wf_tick: %d/%d\n", tick, tot_tick);
    #endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: once the processing is completed and all the units have been
// processed, advance the simulation tick
// ------------------------------------------------------------------------
void wb_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "wb_advance_tick\n");
  #endif

  #ifdef DEBUG
     tot_tick++;
  #endif

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "wb: num_ticks: %d, tick: %d\n", num_ticks, tick);
  #endif
  
  // change packet key colour,
  bkpKey ^= SPINN_COLOUR_KEY;
  
  // and check if end of example's BACKPROP phase
  if (tick == SPINN_WB_END_TICK)
  {
    // compute weight deltas after last tick,
    //TODO: should be called or scheduled?
    //w_weight_deltas ();

    // initialize tick for next example
    tick = SPINN_W_INIT_TICK;

    // go to FORWARD phase,
    w_switch_to_fw ();

    // and move to next example
    //TODO: should be called or scheduled?
    w_advance_example ();
  }
  else
  {
    // if not decrement tick,
    tick--;

    // and trigger computation
    spin1_schedule_callback (wb_process, NULL, NULL, SPINN_WB_PROCESS_P);

    #ifdef TRACE
      io_printf (IO_BUF, "wb_tick: %d/%d\n", tick, tot_tick);
    #endif
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
  
  // check if done with events -- end of example's FORWARD phase
  if (++evt >= num_events)
  {
    // access synchronization semaphore with interrupts disabled
    uint cpsr = spin1_int_disable ();
    
    // initialize synchronization semaphore,
    wf_thrds_done = 0;  // no processing and no stop in tick 0 
    
    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // initialize crit for next example,
    // first tick does not get a stop packet!
    tick_stop = FALSE;

    // and check if in training mode
    if (mlpc.training)
    {
      // if training, save number of ticks
      num_ticks = tick;

      // then do BACKPROP phase
      w_switch_to_bp ();
    }
    else
    {
      // if not training initialize tick for next example,
      tick = SPINN_W_INIT_TICK;

      // and move to next example
      w_advance_example ();
    }
  }
  else
  {
    // if not increment tick,
    tick++;

    #ifdef DEBUG
      tot_tick++;
    #endif

    // change packet key colour,
    fwdKey ^= SPINN_COLOUR_KEY;

    // and trigger computation
    spin1_schedule_callback (wf_process, NULL, NULL, SPINN_WF_PROCESS_P);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// update the example at the end of a simulation tick
// ------------------------------------------------------------------------
void w_advance_example (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "w_advance_example\n");
  #endif
  
  // check if done with examples
  if (++example >= mlpc.num_examples)
  {
    // if training update weights at end of epoch
    if (mlpc.training)
    {
      //TODO: should be called or scheduled?
      w_weight_deltas ();
      w_update_weights ();
      
      #if WEIGHT_HISTORY == TRUE
        // send weight history to host
        //TODO: write this function!
        //send_weights_to_host ();
      #endif
    }

    // check if done with epochs
    if (++epoch >= mlpc.num_epochs)
    {
      // if done then finish
      spin1_stop ();
      return;
    }
    else
    {
      // if not start from first example again,
      example = 0;

      // and, if training, initialize weight changes
      //TODO: find a better place for this operation
      if (mlpc.training)
      {
        for (uint i = 0; i < wcfg.num_rows; i++)
        {
          for (uint j = 0; j < wcfg.num_cols; j++)
          {
            w_wchanges[i][j] = 0;
          }
        }
      }
    }
  }

  // start from first event for next example,
  evt = 0;
  num_events = ex[example].num_events;

  // and send sync packet to allow unit outputs to be sent
  while (!spin1_send_mc_packet (wf_sync_key, 0, NO_PAYLOAD));

  #ifdef DEBUG
    spk_sent++;
  #endif
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

  // and trigger BACKPROP computation
  spin1_schedule_callback (wb_process, NULL, NULL, SPINN_WB_PROCESS_P);

  // and send sync packet to allow unit outputs to be sent
//#  while (!spin1_send_mc_packet (wb_sync_key, 0, NO_PAYLOAD));

//#  #ifdef DEBUG
//#    spk_sent++;
//#  #endif
}
// ------------------------------------------------------------------------
