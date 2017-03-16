// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "sdram.h"

#include "init_i.h"
#include "comms_i.h"
#include "process_i.h"
#include "activation.h"

// set of routines to be used by I core to process data

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
extern uint         event_idx;    // index into current event
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
extern short_weight_t       *wt; //# initial connection weights
extern mlp_set_t            *es; // example set data
extern mlp_example_t        *ex; // example data
extern mlp_event_t          *ev; // event data
extern short_activ_t        *it; // example inputs
extern short_activ_t        *tt; // example targets
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern global_conf_t  mlpc;       // network-wide configuration parameters
extern chip_struct_t  ccfg;       // chip configuration parameters
extern i_conf_t   icfg;           // input core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// input core variables
// ------------------------------------------------------------------------
extern long_net_t     * i_nets;        // unit nets computed in current tick
extern long_delta_t   * i_deltas;      // deltas computed in current tick
extern long_delta_t   * i_init_delta;  // deltas computed in first tick
extern pkt_queue_t      i_pkt_queue;   // queue to hold received b-d-ps
extern uchar            i_active;      // processing b-d-ps from queue?
extern uint             i_it_idx;      // index into current inputs/targets
extern scoreboard_t   * if_arrived;    // keep track of expected net b-d-p
extern scoreboard_t     if_done;       // current tick net computation done
extern uint             if_thrds_done; // sync. semaphore: proc & stop
extern long_delta_t   * ib_init_delta; // initial delta value for every tick
extern scoreboard_t     ib_all_arrived;// all deltas have arrived in tick
extern scoreboard_t   * ib_arrived;    // keep track of expected delta b-d-p
extern scoreboard_t     ib_done;       // current tick delta computation done
//#extern uint             ib_thrds_done; // sync. semaphore: proc & stop
extern long_net_t     * i_last_integr_net; //last integrator output value
extern long_delta_t   * i_last_integr_delta; //last integrator delta value
// list of input pipeline procedures
extern in_proc_t const  i_in_procs[SPINN_NUM_IN_PROCS];
extern in_proc_back_t const  i_in_back_procs[SPINN_NUM_IN_PROCS];
extern long_net_t     * i_net_history; //sdram pointer where to store input history
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
// process queued packets until queue empty
// ------------------------------------------------------------------------
void i_process (uint null0, uint null1)
{  
  #ifdef TRACE
    io_printf (IO_BUF, "i_process\n");
  #endif
  
  // process packet queue
  // access queue with interrupts disabled
  uint cpsr = spin1_int_disable ();

  // process until queue empty
  while (i_pkt_queue.head != i_pkt_queue.tail)
  {
    // if not empty dequeue packet,
    uint key = i_pkt_queue.queue[i_pkt_queue.head].key;
    uint payload = i_pkt_queue.queue[i_pkt_queue.head].payload;
    i_pkt_queue.head = (i_pkt_queue.head + 1) % SPINN_SUM_PQ_LEN;

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

      i_forward_packet (key, payload);
    }
    else
    {
      // process BACKPROP phase packet
      #ifdef DEBUG
        recv_bkp++;
        if (phase != SPINN_BACKPROP)
          wrng_phs++;
      #endif

      i_backprop_packet (key, payload);
    }

    // access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // when done, flag that going to sleep,
  i_active = FALSE;

  // restore interrupts and leave
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process FORWARD phase: apply input pipeline elements
// ------------------------------------------------------------------------
void i_forward_packet (uint key, uint payload)
{
  // get net index: mask out block, phase and colour data,
  uint inx = key & SPINN_NET_MASK;

  // accumulate new net b-d-p,
  // s40.23
  i_nets[inx] = (long_net_t) ((net_t) payload);

  // mark net b-d-p as arrived,
  #if SPINN_USE_COUNTER_SB == FALSE
    // get net block: mask out phase, colour and net index data
    uint blk = (key & SPINN_BLK_R_MASK) >> SPINN_BLK_R_SHIFT;

    // check if already marked -- problem,
    #ifdef DEBUG
      if (if_arrived[inx] & (1 << blk))
      {
        io_printf (IO_BUF, "!c:%u b:%u k:%u a:0x%08x\n",
                    blk, inx, sf_arrived[inx]
                  );
      }
    #endif

      // mark it
    if_arrived[inx] |= (1 << blk);
  #else
    if_arrived[inx]++;
  #endif

  // and check if dot product complete to compute net
  if (if_arrived[inx] == icfg.f_all_arrived)
  {
    net_t net_tmp;

    // compute unit input,
    // TODO: need to make sure this is the same as Lens
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

    // prepare for next tick,
    i_nets[inx] = 0;
    if_arrived[inx] = 0;

    // mark net as done,
    #if SPINN_USE_COUNTER_SB == FALSE
      if_done |= (1 << inx);
    #else
      if_done++;
    #endif

    // and check if all nets done
    if (if_done == icfg.f_all_done)
    {
      // access synchronization semaphore with interrupts disabled
      uint cpsr = spin1_int_disable ();

      // check if all threads done
      if (if_thrds_done == 0)
      {
        // if done initialize semaphore,
        if_thrds_done = 1;

        // restore interrupts after flag access,
        spin1_mode_restore (cpsr);

        // and advance tick
        //TODO: check if need to schedule or can simply call
        if_advance_tick (NULL, NULL);
      }
      else
      {
        // if not done report processing thread done,
        if_thrds_done -= 1;

        // and restore interrupts after flag access
        spin1_mode_restore (cpsr);
      }
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process BACKPROP phase: apply BACKPROP input pipeline elements
// ------------------------------------------------------------------------
void i_backprop_packet (uint key, uint payload)
{
  // get delta index: mask out block, phase and colour data,
  uint inx = key & SPINN_DELTA_MASK;

  // accumulate new delta b-d-p,
  i_deltas[inx] = ((long_delta_t) ((delta_t) payload))
    << (SPINN_LONG_DELTA_SHIFT - SPINN_DELTA_SHIFT);

  // mark delta b-d-p as arrived,
  #if SPINN_USE_COUNTER_SB == FALSE
    // get delta block: mask out phase, colour and net index data
    uint blk = (key & SPINN_BLK_C_MASK) >> SPINN_BLK_C_SHIFT;
  
    // check if already marked -- problem,
    #ifdef DEBUG
      if (ib_arrived[inx] & (1 << blk))
      {
        io_printf (IO_BUF, "!c:%u b:%u k:%u a:0x%08x\n",
                blk, inx, ib_arrived[inx]
                  );
      }
    #endif
 
      // mark it
    ib_arrived[inx] |= (1 << blk);
  #else
    ib_arrived[inx]++;
  #endif

  // and check if delta complete to send to next stage
  // TODO: can use a configuration constant -- needs fixing
  // this core always receives packets from a single S core. Therefore the
  // ib_all_arrived will always be 1, and the if lose meaning: this routine
  // is executed when a packet is received -- to be chekced
  if (ib_arrived[inx] == ib_all_arrived)
  {
    // restore net for the previous tick
    restore_nets (inx, tick -1);

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
    while (!spin1_send_mc_packet ((bkpKey | inx), delta, WITH_PAYLOAD));

    #ifdef DEBUG
      pkt_sent++;
      sent_bkp++;
    #endif

    // prepare for next tick,
    i_deltas[inx] = 0;
    ib_arrived[inx] = 0;

    // mark delta as done,
    #if SPINN_USE_COUNTER_SB == FALSE
      ib_done |= (1 << inx);
    #else
      ib_done++;
    #endif

    // and check if all deltas done
    if (ib_done == icfg.b_all_done)
    {
      // advance tick
      //TODO: check if need to schedule or can simply call
      ib_advance_tick (NULL, NULL);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: the tick has been completed, move FORWARD to the next tick
// updating the indexes to the events/examples as required
// ------------------------------------------------------------------------
void if_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "if_advance_tick\n");
  #endif
  
  // prepare for next tick,
  if_done = 0;

  #ifdef DEBUG
    tot_tick++;
  #endif

  // and check if end of example's FORWARD phase
  if (tick_stop)
  {
    if_advance_event ();
  }
  else
  {
    // if not done increment tick
    tick++;

    #ifdef TRACE
      io_printf (IO_BUF, "if_tick: %d/%d\n", tick, tot_tick);
    #endif
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// BACKPROP phase: the tick has been completed, move FORWARD to the next tick
// updating the indexes to the events/examples as required
// ------------------------------------------------------------------------
void ib_advance_tick (uint null0, uint null1)
{
  #ifdef TRACE
    io_printf (IO_BUF, "ib_advance_tick\n");
  #endif
  
  // prepare for next tick,
  ib_done = 0;

  #ifdef DEBUG
    tot_tick++;
  #endif

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "ib_advance_tick - tick: %d, num_ticks: %d\n", tick, num_ticks);
  #endif

  // and check if end of BACKPROP phase
  if (tick == SPINN_IB_END_TICK)
  {
    // initialize the tick count
    tick = SPINN_I_INIT_TICK;

    #ifdef TRACE
      io_printf (IO_BUF, "w_switch_to_fw\n");
    #endif
  
    // switch to FORWARD phase,
    phase = SPINN_FORWARD;

    // and move to next example
    i_advance_example ();
  }
  else
  {
    // if not done decrement tick
    tick--;

    #ifdef TRACE
      io_printf (IO_BUF, "ib_tick: %d/%d\n", tick, tot_tick);
    #endif
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
  
  // check if done with events
  if (++evt >= num_events)
  {
    // and check if in training mode
    if (mlpc.training)
    {
      // if training, save number of ticks
      num_ticks = tick;

      #ifdef TRACE
        io_printf (IO_BUF, "w_switch_to_bp\n");
      #endif
  
      // then do BACKPROP phase
      phase = SPINN_BACKPROP;
    }
    else
    {
      // if not training, initialize ticks for the next example
      tick = SPINN_I_INIT_TICK;
      // then move to next example
      i_advance_example ();
    }
  }
  else
  {
    // if input or output group update input/target index
    // TODO: to check if the target value is required in I cores
    // for the BACKPROP phase, otherwise remove the condition for the
    // output group
    if (icfg.input_grp || icfg.output_grp)
    {
      i_it_idx += icfg.num_nets;
    }

    // and increment tick
    tick++;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase: update the example at the end of a simulation tick
// ------------------------------------------------------------------------
void i_advance_example (void)
{
  #ifdef TRACE
    io_printf (IO_BUF, "i_advance_example\n");
  #endif
  
  // check if done with examples
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
  //TODO: check if the target value is required in I cores
  // for the BACKPROP phase, otherwise remove the condition for the
  // output group
  if (icfg.input_grp || icfg.output_grp)
  {
    i_it_idx = ev[event_idx].it_idx * icfg.num_nets;
  }

  // if the input integrator is used reset the array of last values
  if (icfg.in_integr_en)
    for (uint i = 0; i < icfg.num_nets; i++)
    {
      i_last_integr_net[i] = (long_net_t) icfg.initNets;
      i_last_integr_delta[i] = 0;
    }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// FORWARD phase:
// call the elements in the input pipeline, as they have been
// specified through splens
// ------------------------------------------------------------------------
void compute_in (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "compute_in\n");
  #endif

  #ifdef DEBUG_VRB
    char* group;
    group = (icfg.input_grp) ? "Input" : ((icfg.output_grp) ? "Output" : ((icfg.num_nets == 1) ? "Bias" : "Hidden"));
    io_printf (IO_BUF, "compute_in - Group: %s - Example: %d - Tick: %d\n", group, example, tick);
  #endif
  
  for (uint i = 0; i < icfg.num_in_procs; i++)
  {
    i_in_procs[icfg.procs_list[i]] (inx);
  }

  // check if in training mode, and if so, store nets
  // TODO: for non-continuous networks, this needs to check the requirement to have these 
  // histories saved, which needs to come from splens. For continuous networks, these histories
  // are always required. 
  if (mlpc.training)
  {
    store_nets(inx);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stores nets for the current tick
// ------------------------------------------------------------------------
void store_nets (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "store_nets\n");
  #endif

  i_net_history[(tick * icfg.num_nets) + inx] = i_nets[inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// restores the net of the specified unit for the requested tick
// ------------------------------------------------------------------------
void restore_nets (uint inx, uint tick)
{
  #ifdef TRACE
    io_printf (IO_BUF, "restore_nets\n");
  #endif

  i_nets[inx] = i_net_history[(tick * icfg.num_nets) + inx];
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// input integrator element
// ------------------------------------------------------------------------
void in_integr (uint inx)
{  
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "in_integr\n");
  #endif
  
  // representation: s40.23 in a 64 bit variable
  long_net_t last_net = i_last_integr_net[inx];
  // representation: s40.23 in a 64 bit variable
  long_net_t desired_net = i_nets[inx];
  // representation: 48.16 in a 64 bit variable
  long long  dt = icfg.in_integr_dt;
  
  // compute the new value of the net as indicated by lens
  // representation: 40.23 + (48.16 * ( 40.23 - 40.23) >> 16) = 40.23
  // all the variables are expanded to 64 bits to avoid overflows and wrap-around
  long_net_t net = last_net + (dt * (desired_net - last_net) >> 16);

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


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
 * Lens computation of the input integrator function
 * 
 * static void integrateInput(Group G, GroupProc P) {
 *   real dt = Net->dt * G->dtScale, *lastInput = P->unitData;
 *   FOR_EACH_UNIT2(G, {
 *     lastInput[u] += dt * U->dtScale * (U->input - lastInput[u]);
 *     U->input = lastInput[u];
 *   });
 * }
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


// ------------------------------------------------------------------------
//soft clamp element
// ------------------------------------------------------------------------
void in_soft_clamp (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "in_soft_clamp\n");
  #endif

  long_activ_t external_input = it[i_it_idx + inx];  // 49.15 repr.

  // compute only if input is not NaN
  if (external_input != (long_activ_t) SPINN_SHORT_ACTIV_NaN)
  {
    lfpreal soft_clamp_strength = icfg.soft_clamp_strength; // 48.16 repr.
    long_activ_t init_output = icfg.initOutput;             // 49.15 repr.
  
    // computation of the soft clamp operator following Lens code
    // representation: 36.27 + (48.16 * (49.15 - 49.15) >> (16 - 12)) = 36.27
    long_activ_t output = init_output
                             + ((soft_clamp_strength
                                 * (external_input - init_output))
                                   >>  (SPINN_FPREAL_SHIFT - SPINN_ACTIV_SHIFT + SPINN_SHORT_ACTIV_SHIFT)
                               );
  
    i_nets[inx] += inv_sigmoid((short_activ_t) (output << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT)));
  }
}
// ------------------------------------------------------------------------


/******************************************************************************/
/*  LENS code starts                                                          */
/******************************************************************************/
/*
 * Lens computation of the soft clamp
 * static void softClampInput(Group G, GroupProc P) {
 *   real initOutput = chooseValue(G->initOutput, Net->initOutput),
 *     gain = chooseValue(G->gain, Net->gain),
 *     strength = chooseValue(G->clampStrength, Net->clampStrength), val;
 *   FOR_EACH_UNIT(G, {
 *     if (!isNaN(U->externalInput)) {
 *       val = initOutput + strength * (U->externalInput - initOutput);
 *       U->input += INV_SIGMOID(val, gain);
 *     }
 *   });
 * }
*/
/******************************************************************************/
/*  LENS code end                                                             */
/******************************************************************************/


// ------------------------------------------------------------------------
// routine which computes the BACKPROP phase of the computation of the
// input elements pipeline
// ------------------------------------------------------------------------
void compute_in_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "compute_in_back\n");
  #endif

  #ifdef DEBUG_VRB
    char* group;
    group = (icfg.input_grp) ? "Input" : ((icfg.output_grp) ? "Output" : ((icfg.num_nets == 1) ? "Bias" : "Hidden"));
    io_printf (IO_BUF, "compute_in_back - Group: %s - Example: %d - Tick: %d\n", group, example, tick);
  #endif
  
  int i;
  
  // the set of procedures needs to be executed in the reverse order, starting
  // from the last input pipeline element, and executing the routine only if the
  // element in the list is not NULL
  if (icfg.num_in_procs >= 1)
    for (i = icfg.num_in_procs-1; i >= 0; i--)
    {
      if (i_in_back_procs[icfg.procs_list[i]] != NULL);
        i_in_back_procs[icfg.procs_list[i]] (inx);
    }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// compute the input integration operation for the backprop
// ------------------------------------------------------------------------
void in_integr_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "in_integr_back\n");
  #endif

  // s36.27
  long_delta_t last_delta = i_last_integr_delta[inx];

  // s47.16
  lfpreal dt = icfg.in_integr_dt;

  // s36.27 = (s47.16 * s36.27) >> 16
  long_delta_t d = (dt * last_delta) >> SPINN_FPREAL_SHIFT;

  // s36.27 = s36.27 + s36.27 - s36.27
  last_delta += i_deltas[inx] - d;

  i_deltas[inx] = d;
  
  // store the integrator state for the next iteration
  i_last_integr_delta[inx] = last_delta;
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
/* There is no softClampInputBack in Lens*/
/*
void in_soft_clamp_back (uint inx)
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "in_soft_clamp_back\n");
  #endif
}
*/
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// initialization of the input intergrator state
// ------------------------------------------------------------------------
int init_in_integr ()
{
  #ifdef TRACE_VRB
    io_printf (IO_BUF, "init_in_integr\n");
  #endif
  
  int i;

  // allocate the memory for the integrator state variable for outputs
  if ((i_last_integr_net = ((long_net_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_net_t)))) == NULL
       )
  {
      return (SPINN_MEM_UNAVAIL);
  }

  // allocate the memory for the integrator state variable for deltas
  if ((i_last_integr_delta = ((long_delta_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_delta_t)))) == NULL
       )
  {
      return (SPINN_MEM_UNAVAIL);
  }

  // reset the memory of the integrator state variable
  for (i = 0; i<icfg.num_nets; i++)
  {
    i_last_integr_net[i] = (long_net_t) icfg.initNets;
    i_last_integr_delta[i] = 0;
  }

  return SPINN_NO_ERROR;
}
// ------------------------------------------------------------------------
