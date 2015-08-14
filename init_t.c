// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "sdram.h"

#include "comms_t.h"
#include "process_t.h"

// this files contains the initialization routine for T cores
// the first routine of the mlp run-time code is scheduled to be executed here

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint coreID;               // 5-bit virtual core ID
extern uint coreIndex;            // coreID - 1 (convenient for array indexing)
extern uint fwdKey;               // 32-bit packet ID for FORWARD phase
extern uint bkpKey;               // 32-bit packet ID for BACKPROP phase
extern uint stpKey;               // 32-bit packet ID for stop criterion

extern uint coreType;             // weight, sum or threshold

extern uint         example;      // current example in epoch
extern uint         num_events;   // number of events in current example
extern uint         event_idx;    // index into current event
extern uint         num_ticks;    // number of ticks in current event
extern uint         max_ticks;    // maximum number of ticks in current event
extern uint         min_ticks;    // minimum number of ticks in current event
extern uint         tick;         // current tick in phase
extern uint         ev_tick;      // current tick in event

extern chip_struct_t        *ct; // chip-specific data
extern uint                 *cm; // simulation core map
extern uchar                *dt; // core-specific data
extern mc_table_entry_t     *rt; // multicast routing table data
extern weight_t             *wt; // initial connection weights
extern struct mlp_set       *es; // example set data
extern struct mlp_example   *ex; // example data
extern struct mlp_event     *ev; // event data
extern activation_t         *it; // example inputs
extern activation_t         *tt; // example targets

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern global_conf_t  mlpc;       // network-wide configuration parameters
extern chip_struct_t  ccfg;       // chip configuration parameters
extern t_conf_t       tcfg;       // threshold core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// threshold core variables
// ------------------------------------------------------------------------
extern activation_t   * t_outputs;     // current tick unit outputs
extern net_t          * t_nets;        // nets received from sum cores
extern error_t        * t_errors[2];   // error banks: current and next tick
extern activation_t   * t_last_integr_output;   //last integrator output value
extern llong_deriv_t  * t_last_integr_output_deriv; //last integrator output deriv value
extern uchar            t_hard_clamp_en; //hard clamp output enabled
extern uint             t_it_idx;      // index into current inputs/targets
extern uint             t_tot_ticks;   // total ticks on current example
extern pkt_queue_t      t_net_pkt_q;   // queue to hold received nets
extern uchar            t_active;      // processing nets/errors from queue?
extern scoreboard_t     t_sync_arr;    // keep track of expected sync packets
extern uchar            t_sync_done;   // have expected sync packets arrived?
extern sdp_msg_t        t_sdp_msg;     // SDP message buffer for host comms.
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
extern uint             tb_comms;      // pointer to receiving errors
extern scoreboard_t     tb_arrived;    // keep track of expected errors
extern uint             tb_thrds_done; // sync. semaphore: proc & stop
extern int              t_max_output_unit; // unit with highest output
extern int              t_max_target_unit; // unit with highest target
extern activation_t     t_max_output;      // highest output value
extern activation_t     t_max_target;      // highest target value
// list of output pipeline procedures
extern out_proc_t const  t_out_procs[SPINN_NUM_OUT_PROCS];
// list of stop eval procedures
extern stop_crit_t const t_stop_procs[SPINN_NUM_STOP_PROCS];
// list of initialization procedures for output pipeline
extern out_proc_init_t const t_init_out_procs[SPINN_NUM_OUT_PROCS];
// derivative of the output
extern llong_deriv_t  * t_output_deriv;
// history arrays
extern activation_t   * t_output_history;
extern llong_deriv_t  * t_output_deriv_history;
extern delta_t        * t_deltas;
extern activation_t   * t_target_history;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
#ifdef DEBUG
  extern uint pkt_sent;  // total packets sent
  extern uint sent_fwd;  // packets sent in FORWARD phase
#endif
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// allocate memory and initialize variables
// ------------------------------------------------------------------------
uint t_init (void)
{
  uint i = 0;
  
  // allocate memory for nets -- stored in FORWARD phase for use in BACKPROP
  if ((t_nets = ((net_t *)
         spin1_malloc (tcfg.num_outputs * sizeof(net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for outputs
  if ((t_outputs = ((activation_t *)
         spin1_malloc (tcfg.num_outputs * sizeof(activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for output derivative (which is equal to error derivative)
  if ((t_output_deriv = ((llong_deriv_t *)
         spin1_malloc (tcfg.num_outputs * sizeof(llong_deriv_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for deltas
  if ((t_deltas = ((delta_t *)
	 spin1_malloc (tcfg.num_outputs * sizeof(delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }
  
  // allocate memory for errors
  if ((t_errors[0] = ((error_t *)
         spin1_malloc (tcfg.num_outputs * sizeof(error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((t_errors[1] = ((error_t *)
         spin1_malloc (tcfg.num_outputs * sizeof(error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // initialize deltas
  for (i = 0; i < tcfg.num_outputs; i++)
  {
    t_deltas[i] = 0;
  }

  // check if the hard clamp is in use in the sequence of pipeline elements
  t_hard_clamp_en = FALSE;
  for (i = 0; i < tcfg.num_out_procs; i++)
  {    
    // check if the hard clamp is in the output pipeline
    // and set hard_clamp_en appropriately
    if (t_out_procs[tcfg.procs_list[i]] == out_hard_clamp)
      t_hard_clamp_en = TRUE;
  }

  // allocate memory for net packet queue
  // TODO: use correct length!
  if ((t_net_pkt_q.queue = ((packet_t *)
         spin1_malloc (SPINN_THLD_PQ_LEN * sizeof(packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // if the network requires training and elements of the pipeline require
  // initialization, then follow the appropriate procedure
  // use the list of procedures in use from lens and call the appropriate
  // initialization routine from the t_init_out_procs function pointer list
  for (i = 0; i < tcfg.num_out_procs; i++)
    if (t_init_out_procs[tcfg.procs_list[i]] != NULL)
    {
      int return_value;
      // call the appropriate routine for pipeline initialization
      return_value = t_init_out_procs[tcfg.procs_list[i]]();
      
      // if return value contains error, return it
      if (return_value != SPINN_NO_ERROR)
        return return_value;
    }
  
  // intialize example and event ticks
  tick = SPINN_T_INIT_TICK;
  ev_tick = SPINN_T_INIT_TICK;

  // initialize max and min ticks
  if (tcfg.is_last_output_group)
  {
    // get max number of ticks for first event
    if (ev[event_idx].max_time != SPINN_FP_NaN)
      max_ticks = (ev[event_idx].max_time * mlpc.ticks_per_int)
                    >> SPINN_FPREAL_SHIFT;
    else
      max_ticks = (es->max_time * mlpc.ticks_per_int) >> SPINN_FPREAL_SHIFT;

    // get min number of ticks for first event
    if (ev[event_idx].min_time != SPINN_FP_NaN)
      min_ticks = (ev[event_idx].min_time * mlpc.ticks_per_int)
                    >> SPINN_FPREAL_SHIFT;
    else
      min_ticks = (es->min_time * mlpc.ticks_per_int) >> SPINN_FPREAL_SHIFT;
  }

  // initialize pointers to received errors
  tb_procs = 0;
  tb_comms = 1;

  // initialize synchronization semaphores
  tb_thrds_done = 1;

  // initialize received nets and errors scoreboard
  tf_arrived = 0;
  tb_arrived = 0;

  // initialize synchronization semaphores
  //TODO: why is this necessary?
  if (tcfg.is_last_output_group)
    tf_thrds_init = 1;  //##
  else
    tf_thrds_init = 1;

  tf_thrds_done = tf_thrds_init;

  // initialize stop flags and function
  if (tcfg.output_grp)
  {
    // variables for stop criterion computation
    t_max_output_unit = -1;
    t_max_target_unit = -1;
    t_max_output = SPINN_ACTIV_MIN;
    t_max_target = SPINN_ACTIV_MIN;

    // no need to wait for previous if first in chain
    if (tcfg.is_first_output_group)
      tf_stop_init = 0;
    else
      tf_stop_init = 1;

    tf_stop_done = tf_stop_init;
    tf_stop_prev = TRUE;
    tf_stop_crit = TRUE;

    tf_stop_func = t_stop_procs[tcfg.criterion_function];

    if (tcfg.is_last_output_group)
    {
      // "broadcast" key
      tf_stop_key = SPINN_STPR_KEY | SPINN_TB_KEY(tcfg.output_blk);
    }
    else
    {
      // "daisy chain" key
      tf_stop_key = SPINN_STPF_KEY | SPINN_TB_KEY(tcfg.output_blk);
    }
  }

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "tsk = 0x%08x\n", tf_stop_key);
  #endif

  // initialize processing thread flag
  t_active = FALSE;

  // initialize received sync packets scoreboard
  t_sync_arr = 0;

  // initialize sync packets flag
  t_sync_done = FALSE;

  // initialize net packet queue
  t_net_pkt_q.head = 0;
  t_net_pkt_q.tail = 0;

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "wo:%d\n", tcfg.write_out);
  #endif
  
  // check if writing outputs to host
  if (tcfg.write_out)
  {
    // initialize SDP message buffer
    // Fill in SDP destination fields
    t_sdp_msg.tag = SPINN_SDP_IPTAG;      // IPTag
    t_sdp_msg.dest_port = PORT_ETH;       // Ethernet
    t_sdp_msg.dest_addr = sv->dbg_addr;   // Root chip
    
    // Fill in SDP source & flag fields
    t_sdp_msg.flags = SPINN_SDP_FLAGS;
    t_sdp_msg.srce_port = coreID;
    t_sdp_msg.srce_addr = sv->p2p_addr;
    
    // compute total ticks in first example
    //TODO: cannot compute correctly -- variable if completion criteria used
    t_tot_ticks = 0;
    for (uint i = 0; i < num_events; i++)
    {
      // update number of ticks for new event
      if (ev[event_idx + i].max_time != SPINN_FP_NaN)
      {
        t_tot_ticks += (ev[event_idx + i].max_time * mlpc.ticks_per_int)
                         >> SPINN_FPREAL_SHIFT;
      }
      else
      {
        t_tot_ticks += (es->max_time * mlpc.ticks_per_int)
                         >> SPINN_FPREAL_SHIFT;
      }
    }

    // schedule sending of initial data to host
    spin1_schedule_callback (send_info_to_host, NULL, NULL, SPINN_T_INIT_OUT_P);
  }
  
  // initialize packet keys
  //NOTE: colour is initialized to 0
  fwdKey = SPINN_TB_KEY(tcfg.output_blk) | SPINN_CORETYPE_KEY
             | SPINN_PHASE_KEY(SPINN_FORWARD);

  bkpKey = SPINN_TB_KEY(tcfg.delta_blk)  | SPINN_CORETYPE_KEY
             | SPINN_PHASE_KEY(SPINN_BACKPROP);

  // if input or output group initialize event input/target index
  if (tcfg.input_grp || tcfg.output_grp)
  {
    t_it_idx = ev[event_idx].it_idx * tcfg.num_outputs;
  }

  // TODO: the following memory allocation is to be used to store
  // the history of any of these three sets of values. When training
  // continuous networks, these three histories always need to be saved.
  // For non-continuous networks, they only need to be stored if the 
  // backpropTicks field of the network is greater than one. This
  // information needs to come from splens in the tcfg structure.

  // allocate memory in SDRAM for output history
  if ((t_output_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_outputs * mlpc.global_max_ticks * sizeof(activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }
  
  // allocate memory in SDRAM for target history
  if ((t_target_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_outputs * mlpc.global_max_ticks * sizeof(activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for output derivative history
  if ((t_output_deriv_history = ((llong_deriv_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_outputs * mlpc.global_max_ticks * sizeof(llong_deriv_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // schedule initialization and sending of unit outputs
  spin1_schedule_callback (t_init_outputs, NULL, NULL, SPINN_T_INIT_OUT_P);
  spin1_schedule_callback (send_outputs_to_host,
                            SPINN_HOST_NORMAL, 0, SPINN_SEND_OUTS_P
                          );

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
