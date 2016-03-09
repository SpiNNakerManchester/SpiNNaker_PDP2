// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "sdram.h"

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"

#define SPINN_EXEC_TYPE 'T'

// main methods for the T core

// ------------------------------------------------------------------------
// global "constants"
// ------------------------------------------------------------------------

// list of procedures for the FORWARD phase in the output pipeline. The order is
// relevant, as the index is defined in mlp_params.h
out_proc_t const
  t_out_procs[SPINN_NUM_OUT_PROCS] = 
  {
    out_logistic, out_integr, out_hard_clamp, out_weak_clamp, out_bias
  };

// list of procedures for the BACKPROP phase in the output pipeline. The order
// is relevant, as the index needs to be the same as in the FORWARD phase. In
// case one routine is not intended to be available in lens, then a NULL should
// replace the call
out_proc_back_t const
  t_out_back_procs[SPINN_NUM_OUT_PROCS] = 
  {
    out_logistic_back, out_integr_back, out_hard_clamp_back, out_weak_clamp_back, out_bias_back
  };
  
// list of procedures for the initialization of the output pipeline. The order
// is relevant, as the index needs to be the same as in the FORWARD phase. In
// case one routine is not intended to be available because no initialization
// is required, then a NULL should replace the call
out_proc_init_t const
  t_init_out_procs[SPINN_NUM_OUT_PROCS] = 
  {
      NULL, init_out_integr, init_out_hard_clamp, init_out_weak_clamp, NULL
  };

// list of procedures for the evaluation of the convergence (and stopping)
// criteria. The order is relevant, as the indexes are specified in mlp_params.h
// A NULL routine does not evaluate any convergence criterion and therefore the
// simulation is always performed for the maximum number of ticks
stop_crit_t const
  t_stop_procs[SPINN_NUM_STOP_PROCS] = 
  {
    NULL, std_stop_crit, max_stop_crit
  };
  
// list of procedures for the evaluation of the errors between the output and
// the target values of the output groups. The order is relevant, as the indexes
// are specified in mlp_params.h. A NULL routine does not evaluate any error and
// therefore the weight update will always be 0
out_error_t const
  t_out_error[SPINN_NUM_ERROR_PROCS] = 
  {
    NULL, error_cross_entropy, error_squared
  };
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
uint chipID;               // 16-bit (x, y) chip ID
uint coreID;               // 5-bit virtual core ID
uint coreIndex;            // coreID - 1 (convenient for array indexing)
uint fwdKey;               // 32-bit packet ID for FORWARD phase
uint bkpKey;               // 32-bit packet ID for BACKPROP phase
uint stpKey;               // 32-bit packet ID for stop criterion

uint coreType;             // weight, sum or threshold

uint         epoch;        // current training iteration
uint         example;      // current example in epoch
uint         evt;          // current event in example
uint         num_events;   // number of events in current example
uint         event_idx;    // index into current event
proc_phase_t phase;        // FORWARD or BACKPROP
uint         num_ticks;    // number of ticks in current event
uint         max_ticks;    // maximum number of ticks in current event
uint         min_ticks;    // minimum number of ticks in current event
uint         tick;         // current tick in phase
uint         ev_tick;      // current tick in event
uchar        tick_stop;    // current tick stop decision

// ------------------------------------------------------------------------
// configuration structures (SDRAM)
// ------------------------------------------------------------------------
uint             *cm; // simulation core map
chip_struct_t    *ct; // chip-specific data
uchar            *dt; // core-specific data
mc_table_entry_t *rt; // multicast routing table data
short_weight_t   *wt; // initial connection weights
mlp_set_t        *es; // example set data
mlp_example_t    *ex; // example data
mlp_event_t      *ev; // event data
short_activ_t    *it; // example inputs
short_activ_t    *tt; // example targets

// ------------------------------------------------------------------------
// network and core configurations (DTCM)
// ------------------------------------------------------------------------
global_conf_t mlpc;           // network-wide configuration parameters
chip_struct_t ccfg;           // chip configuration parameters
t_conf_t      tcfg;           // threshold core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// threshold core variables
// ------------------------------------------------------------------------
// threshold cores compute unit outputs and error deltas.
// ------------------------------------------------------------------------
activation_t   * t_outputs;         // current tick unit outputs
net_t          * t_nets;            // nets received from sum cores
error_t        * t_errors[2];       // error banks: current and next tick
activation_t  * t_last_integr_output;  //last integrator output value
long_deriv_t   * t_last_integr_output_deriv; //last integrator output deriv value
activation_t   * t_instant_outputs; // current output value stored for the backward pass
short_activ_t  * t_out_hard_clamp_data; //values injected by hard clamps
short_activ_t  * t_out_weak_clamp_data; //values injected by weak clamps
uchar            t_hard_clamp_en;   //hard clamp output enabled
uint             t_it_idx;          // index into current inputs/targets
uint             t_tot_ticks;       // total ticks on current example
pkt_queue_t      t_net_pkt_q;       // queue to hold received nets
uchar            t_active;          // processing nets/errors from queue?
scoreboard_t     t_sync_arr;        // keep track of expected sync packets
uchar            t_sync_done;       // have expected sync packets arrived?
sdp_msg_t        t_sdp_msg;         // SDP message buffer for host comms.

// FORWARD phase specific
// (output computation)
scoreboard_t     tf_arrived;        // keep track of expected nets
uint             tf_thrds_init;     // sync. semaphore: proc & stop
uint             tf_thrds_done;     // sync. semaphore: proc & stop
uchar            tf_stop_prev;      // previous group stop criterion met?
uchar            tf_stop_crit;      // stop criterion met?
uchar            tf_stop_init;      // sync. semaphore: stop daisy chain
uchar            tf_stop_done;      // sync. semaphore: stop daisy chain
stop_crit_t      tf_stop_func;      // stop evaluation function
uint             tf_stop_key;       // stop criterion packet key

// BACKPROP phase specific
// (error delta computation)
uint             tb_procs;          // pointer to processing errors
uint             tb_comms;          // pointer to receiving errors
scoreboard_t     tb_arrived;        // keep track of expected errors
uint             tb_thrds_done;     // sync. semaphore: proc & stop

int              t_max_output_unit; // unit with highest output
int              t_max_target_unit; // unit with highest target
activation_t     t_max_output;      // highest output value
activation_t     t_max_target;      // highest target value

long_deriv_t  * t_output_deriv;    // derivative of the output value
long_deriv_t  * t_output_deriv_history;
delta_t        * t_deltas;
short_activ_t  * t_target_history;
net_t          * t_net_history;
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
#ifdef DEBUG
  uint pkt_sent = 0;  // total packets sent
  uint sent_fwd = 0;  // packets sent in FORWARD phase
  uint sent_bkp = 0;  // packets sent in BACKPROP phase
  uint pkt_recv = 0;  // total packets received
  uint recv_fwd = 0;  // packets received in FORWARD phase
  uint recv_bkp = 0;  // packets received in BACKPROP phase
  uint spk_sent = 0;  // sync packets sent
  uint spk_recv = 0;  // sync packets received
  uint stp_sent = 0;  // stop packets sent
  uint stp_recv = 0;  // stop packets received
  uint wrng_phs = 0;  // packets received in wrong phase
  uint wrng_tck = 0;  // FORWARD packets received in wrong tick
  uint wrng_btk = 0;  // BACKPROP packets received in wrong tick
  uint wght_ups = 0;  // number of weight updates done
  uint tot_tick = 0;  // total number of ticks executed
#endif
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// load configuration from SDRAM and initialize variables
// ------------------------------------------------------------------------
uint init ()
{
  // return code
  uint rcode = SPINN_NO_ERROR;
  
  // initialize network configuration from SDRAM
  spin1_memcpy (&mlpc, gt, sizeof(global_conf_t));
  
  // initialize chip-specific configuration from SDRAM
  ct = (chip_struct_t *) mlpc.chip_struct_addr;
  spin1_memcpy(&ccfg, ct, sizeof(chip_struct_t));
  
  //initialize pointers to the appropriate structures
  cm = (uint *) ccfg.cm_struct_addr;                // simulation core map
  dt = (uchar *) ccfg.core_struct_addr[coreIndex];  // core-specific data
  
  es = (struct mlp_set *) ccfg.example_set_addr;    // example set data
  ex = (struct mlp_example *) ccfg.examples_addr;   // example data
  ev = (struct mlp_event *) ccfg.events_addr;       // event data
  
  // initialize global stop criteron packet key
  stpKey = SPINN_STPR_KEY;

  #ifdef DEBUG_VRB
    io_printf (IO_BUF, "sk = 0x%08x\n", stpKey);
  #endif

  // initialize epoch, example and event counters
  //TODO: alternative algorithms for chosing example order!
  epoch   = 0;
  example = 0;
  evt     = 0;

  // initialize phase
  phase = SPINN_FORWARD;

  // initialize number of events and event index
  num_events = ex[example].num_events;
  event_idx  = ex[example].ev_idx;

  // initialize core configuration according to core function
  coreType = ccfg.core_type[coreIndex];
  
  if (coreType != SPINN_THRESHOLD_PROC)
    return SPINN_CORE_TYPE_ERROR;
  
  io_printf (IO_STD, "threshold\n");

  spin1_memcpy (&tcfg, dt, sizeof(t_conf_t));

  it = (short_activ_t *) tcfg.inputs_addr;         // example inputs
  tt = (short_activ_t *) tcfg.targets_addr;        // example targets

  // allocate memory and initialize variables,
  rcode = t_init ();

  // if init went well fill routing table -- only 1 core needs to do it
  if (leadAp && (rcode == SPINN_NO_ERROR))
  {
    if (*(uint*)ccfg.rt_struct_addr != ccfg.num_rt_entries)
        io_printf (IO_STD,
                    "Warning: routing table size mismatch - ccfg: %d, rt: %d\n",
                    ccfg.num_rt_entries, *(uint*)ccfg.rt_struct_addr
                  );

    // multicast routing table data: first word is length!
    rt = (mc_table_entry_t *) (ccfg.rt_struct_addr + sizeof (uint));
    
    // allocate space in routing table
    uint e = rtr_alloc (ccfg.num_rt_entries, 0); // allocate router entries
    if (e == 0)
      rt_error (RTE_ABORT);

    // fill the routing tables with the values from the configuration files
    for (uint i = 0; i < ccfg.num_rt_entries; i++)
    {
      rtr_mc_set (e + i,
                   rt[i].key,
                   rt[i].mask,
                   rt[i].route
                 );
    }
  }

  return (rcode);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// check exit code and print details of the state
// ------------------------------------------------------------------------
void done (uint ec)
{
  // skew execution to avoid tubotron congestion
  spin1_delay_us (SPINN_SKEW_DELAY);  //@delay

  // report problems -- if any
  switch (ec)
  {
    case SPINN_NO_ERROR:
      io_printf (IO_STD, "simulation OK\n");

      break;

    case SPINN_UKNOWN_TYPE:
      io_printf (IO_STD, "unknown core type\n");
      io_printf (IO_BUF, "unknown core type\n");

      break;

    case SPINN_QUEUE_FULL:
      io_printf (IO_STD, "packet queue full\n");
      io_printf (IO_BUF, "packet queue full\n");

      break;

    case SPINN_MEM_UNAVAIL:
      io_printf (IO_STD, "malloc failed\n");
      io_printf (IO_BUF, "malloc failed\n");

      break;

    case SPINN_UNXPD_PKT:
      io_printf (IO_STD, "unexpected packet received - abort!\n");
      io_printf (IO_BUF, "unexpected packet received - abort!\n");

      break;

    case SPINN_TIMEOUT_EXIT:
      io_printf (IO_STD, "timeout - see I/O buffer for log - abort!\n");
      io_printf (IO_BUF, "timeout (h:%u e:%u p:%u t:%u) - abort!\n",
                 epoch, example, phase, tick
                );

      #ifdef DEBUG_VRB
        io_printf (IO_BUF, "(tactive:%u ta:0x%08x/0x%08x tb:0x%08x/0x%08x)\n",
                    t_active, tf_arrived, tcfg.f_all_arrived,
                    tb_arrived, tcfg.b_all_arrived
                  );
        io_printf (IO_BUF, "(tsd:%u tsa:0x%08x/0x%08x)\n",
                    t_sync_done, t_sync_arr, tcfg.f_s_all_arr
                  );
      #endif

      if (tcfg.write_out)  // make sure the output monitor closes!
      {
        send_outputs_to_host (SPINN_HOST_FINAL, tick);
      }

      break;

    // in case the chip configuration data structure defines the core to be of a
    // different type than this executable, throw an error
    case SPINN_CORE_TYPE_ERROR:
      
      switch (coreType)
      {
        case SPINN_WEIGHT_PROC:
          io_printf (IO_STD, "error in the core type - executable: %c core, structure: W type\n", SPINN_EXEC_TYPE);
          break;
          
        case SPINN_SUM_PROC:
          io_printf (IO_STD, "error in the core type - executable: %c core, structure: S type\n", SPINN_EXEC_TYPE);
          break;
          
        case SPINN_INPUT_PROC:
          io_printf (IO_STD, "error in the core type - executable: %c core, structure: I type\n", SPINN_EXEC_TYPE);
          break;
          
        case SPINN_THRESHOLD_PROC:
          io_printf (IO_STD, "error in the core type - executable: %c core, structure: T type\n", SPINN_EXEC_TYPE);
          break;
          
        case SPINN_UNUSED_PROC:
          io_printf (IO_STD, "error in the core type - executable: %c core, but the core should be unused\n", SPINN_EXEC_TYPE);
          break;
        
        default:
          io_printf (IO_STD, "error in the core type - executable: %c core, but chip structure has an invalid entry: %d\n", SPINN_EXEC_TYPE, coreType);
          break;
      }
      
      break;
      
  }

  // report diagnostics
  #ifdef DEBUG
    io_printf (IO_BUF, "total ticks:%d\n", tot_tick);
    io_printf (IO_BUF, "recv:%d fwd:%d bkp:%d\n", pkt_recv, recv_fwd, recv_bkp);
    io_printf (IO_BUF, "sent:%d fwd:%d bkp:%d\n", pkt_sent, sent_fwd, sent_bkp);
    io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
    io_printf (IO_BUF, "wrong tick:%d\n", wrng_tck);
    io_printf (IO_BUF, "wrong btick:%d\n", wrng_btk);
    io_printf (IO_BUF, "sync recv:%d\n", spk_recv);
    io_printf (IO_BUF, "sync sent:%d\n", spk_sent);
    io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
    io_printf (IO_BUF, "stop sent:%d\n", stp_sent);
  #endif
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// timer callback: if the execution takes too long it probably deadlocked.
// Therefore the execution is terminated with SPINN_TIMEOUT_EXIT exit code.
// ------------------------------------------------------------------------
void timeout (uint ticks, uint null)
{
  if (ticks == mlpc.timeout)
  {
    // exit and report timeout
    spin1_kill (SPINN_TIMEOUT_EXIT);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// main: register callbacks and initialize basic system variables
// ------------------------------------------------------------------------
void c_main ()
{
  // say hello,
  io_printf (IO_STD, ">> mlp\n");

  // get this core's IDs,
  chipID = spin1_get_chip_id();
  coreID = spin1_get_core_id();
  coreIndex = coreID - 1; // used to access arrays!

  // initialize application,
  uint exit_code = init ();

  // check if init completed successfully,
  if (exit_code != SPINN_NO_ERROR)
  {

    // if init failed report results,
    done (exit_code);

    // and abort simulation
    return;
  }

  // set the core map for the simulation,
  spin1_set_core_map (mlpc.num_chips, cm);

  // set timer tick value (in microseconds),
  spin1_set_timer_tick (SPINN_TIMER_TICK_PERIOD);

  #ifdef PROFILE
    // configure timer 2 for profiling
    // enabled, 32 bit, free running, 16x pre-scaler
    tc[T2_CONTROL] = SPINN_TIMER2_CONF;
    tc[T2_LOAD] = SPINN_TIMER2_LOAD;
  #endif

  // register callbacks,
  // timeout escape -- in case something went wrong!
  spin1_callback_on (TIMER_TICK, timeout, SPINN_TIMER_P);

  // packet received callback depends on core function
  spin1_callback_on (MC_PACKET_RECEIVED, t_receivePacket, SPINN_PACKET_P);

  // go,
  io_printf (IO_STD, "-----------------------\n");
  io_printf (IO_STD, "starting simulation\n");

  #ifdef PROFILE
    uint start_time = tc[T2_COUNT];
    io_printf (IO_STD, "start count: %u\n", start_time);
  #endif

  // start execution and get exit code,
  exit_code = spin1_start ();

  #ifdef PROFILE
    uint final_time = tc[T2_COUNT];
    io_printf (IO_STD, "final count: %u\n", final_time);
    io_printf (IO_STD, "execution time: %u us\n",
                  (start_time - final_time) / SPINN_TIMER2_DIV);
  #endif

  // report results,
  done (exit_code);

  io_printf (IO_STD, "stopping simulation\n");
  io_printf (IO_STD, "-----------------------\n");

  // and say goodbye
  io_printf (IO_STD, "<< mlp\n");
}
// ------------------------------------------------------------------------
