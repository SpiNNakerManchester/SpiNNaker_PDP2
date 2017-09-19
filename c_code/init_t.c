// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_t.h"
#include "process_t.h"

// this files contains the initialization routine for T cores
// the first routine of the mlp run-time code is scheduled to be executed here

// ------------------------------------------------------------------------
// allocate memory and initialize variables
// ------------------------------------------------------------------------
uint t_init (void)
{
  uint i = 0;

  // allocate memory for nets -- stored in FORWARD phase for use in BACKPROP
  if ((t_nets = ((net_t *)
         spin1_malloc (tcfg.num_units * sizeof(net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for outputs
  if ((t_outputs = ((activation_t *)
         spin1_malloc (tcfg.num_units * sizeof(activation_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for output derivative (which is equal to error derivative)
  if ((t_output_deriv = ((long_deriv_t *)
         spin1_malloc (tcfg.num_units * sizeof(long_deriv_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for deltas
  if ((t_deltas = ((delta_t *)
	 spin1_malloc (tcfg.num_units * sizeof(delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((t_errors[0] = ((error_t *)
         spin1_malloc (tcfg.num_units * sizeof(error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((t_errors[1] = ((error_t *)
         spin1_malloc (tcfg.num_units * sizeof(error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // initialize output derivatives
  for (i = 0; i < tcfg.num_units; i++)
  {
    t_output_deriv[i] = 0;
  }

  // initialize deltas
  for (i = 0; i < tcfg.num_units; i++)
  {
    t_deltas[i] = 0;
  }

  // initialize errors
  for (i = 0; i < tcfg.num_units; i++)
  {
    t_errors[0][i] = 0;
    t_errors[1][i] = 0;
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
      max_ticks = (((ev[event_idx].max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;
    else
      max_ticks = (((es->max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                     + (1 << (SPINN_FPREAL_SHIFT - 1)))
                     >> SPINN_FPREAL_SHIFT;

    // get min number of ticks for first event
    if (ev[event_idx].min_time != SPINN_FP_NaN)
      min_ticks = (((ev[event_idx].min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                    + (1 << (SPINN_FPREAL_SHIFT - 1)))
                    >> SPINN_FPREAL_SHIFT;
    else
      min_ticks = (((es->min_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                    + (1 << (SPINN_FPREAL_SHIFT - 1)))
                    >> SPINN_FPREAL_SHIFT;
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
    t_max_output = SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT);
    t_max_target = SPINN_SHORT_ACTIV_MIN << (SPINN_ACTIV_SHIFT - SPINN_SHORT_ACTIV_SHIFT);

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
      tf_stop_key = rt[STP] | SPINN_STPR_KEY;
    }
    else
    {
      // "daisy chain" key
      tf_stop_key = rt[STP] | SPINN_STPF_KEY;
    }
  }

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "tsk = 0x%08x\n", tf_stop_key);
  #endif

  // initialize processing thread flag
  t_active = FALSE;

  // initialize received sync packets scoreboard
  t_sync_arrived = 0;

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
        t_tot_ticks += (((ev[event_idx + i].max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
      }
      else
      {
        t_tot_ticks += (((es->max_time + SPINN_SMALL_VAL) * ncfg.ticks_per_int)
                         + (1 << (SPINN_FPREAL_SHIFT - 1)))
                         >> SPINN_FPREAL_SHIFT;
      }
    }

    if (t_tot_ticks > ncfg.global_max_ticks - 1)
    {
      t_tot_ticks = ncfg.global_max_ticks - 1;
    }

    // schedule sending of initial data to host
    spin1_schedule_callback (send_info_to_host, NULL, NULL, SPINN_T_INIT_OUT_P);
  }

  // initialize packet keys
  //NOTE: colour is initialized to 0
  fwdKey = rt[FWD] | SPINN_PHASE_KEY (SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY (SPINN_BACKPROP);

  // if input or output group initialize event input/target index
  if (tcfg.input_grp || tcfg.output_grp)
  {
    t_it_idx = ev[event_idx].it_idx * tcfg.num_units;
  }

  // TODO: the following memory allocation is to be used to store
  // the history of any of these sets of values. When training
  // continuous networks, these histories always need to be saved.
  // For non-continuous networks, they only need to be stored if the
  // backpropTicks field of the network is greater than one. This
  // information needs to come in the tcfg structure.

  // allocate memory in SDRAM for target history
  if ((t_target_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for output derivative history
  if ((t_output_deriv_history = ((long_deriv_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (long_deriv_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for net history
  if ((t_net_history = ((net_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (net_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory in SDRAM for output history
  if ((t_output_history = ((activation_t *)
          sark_xalloc (sv->sdram_heap,
                       tcfg.num_units * ncfg.global_max_ticks * sizeof (activation_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // schedule initialization and sending of unit outputs
  spin1_schedule_callback (t_init_outputs, NULL, NULL, SPINN_T_INIT_OUT_P);

  // and, if required, send outputs to host
  if (tcfg.write_out)
  {
    spin1_schedule_callback (send_outputs_to_host,
                              SPINN_HOST_NORMAL, 0, SPINN_SEND_OUTS_P
                            );

  }

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
