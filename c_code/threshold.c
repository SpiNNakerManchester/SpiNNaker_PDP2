// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include <data_specification.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "mlp_externs.h"  // allows compiler to check extern types!

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"

// main methods for the threshold core

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

uint fwdKey;               // 32-bit packet ID for FORWARD phase
uint bkpKey;               // 32-bit packet ID for BACKPROP phase

uint         epoch;        // current training iteration
uint         example;      // current example in epoch
uint         evt;          // current event in example
uint         max_evt;      // the last event reached in the current example
uint         num_events;   // number of events in current example
uint         event_idx;    // index into current event
proc_phase_t phase;        // FORWARD or BACKPROP
uint         num_ticks;    // number of ticks in current event
uint         max_ticks;    // maximum number of ticks in current event
uint         min_ticks;    // minimum number of ticks in current event
uint         tick;         // current tick in phase
uint         ev_tick;      // current tick in event
uchar        tick_stop;    // current tick stop decision
uchar        network_stop; // network_stop decision

uint         to_epoch   = 0;
uint         to_example = 0;
uint         to_tick    = 0;

// ------------------------------------------------------------------------
// data structures in regions of SDRAM
// ------------------------------------------------------------------------
mlp_set_t        *es; // example set data
mlp_example_t    *ex; // example data
mlp_event_t      *ev; // event data
activation_t     *it; // example inputs
activation_t     *tt; // example targets
uint             *rt; // multicast routing keys data

// ------------------------------------------------------------------------
// network and core configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t ncfg;           // network-wide configuration parameters
t_conf_t       tcfg;           // threshold core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// threshold core variables
// ------------------------------------------------------------------------
// threshold cores compute unit outputs and error deltas.
// ------------------------------------------------------------------------
activation_t   * t_outputs;         // current tick unit outputs
net_t          * t_nets;            // nets received from sum cores
error_t        * t_errors[2];       // error banks: current and next tick
activation_t   * t_last_integr_output;  //last integrator output value
long_deriv_t   * t_last_integr_output_deriv; //last integrator output deriv value
activation_t   * t_instant_outputs; // current output value stored for the backward pass
short_activ_t  * t_out_hard_clamp_data; //values injected by hard clamps
short_activ_t  * t_out_weak_clamp_data; //values injected by weak clamps
uchar            t_hard_clamp_en;   //hard clamp output enabled
uint             t_it_idx;          // index into current inputs/targets
uint             t_tot_ticks;       // total ticks on current example
pkt_queue_t      t_net_pkt_q;       // queue to hold received nets
uchar            t_active;          // processing nets/errors from queue?
scoreboard_t     t_sync_arrived;    // keep track of expected sync packets
uchar            t_sync_rdy;        // have expected sync packets arrived?
sdp_msg_t        t_sdp_msg;         // SDP message buffer for host comms.

// FORWARD phase specific
// (output computation)
scoreboard_t     tf_arrived;        // keep track of expected nets
uint             tf_thrds_done;     // sync. semaphore: proc & stop
uchar            tf_chain_prev;     // previous daisy chain (DC) value
uchar            tf_chain_init;     // previous DC received init
uchar            tf_chain_rdy;      // local DC value can be forwarded
uchar            tf_stop_crit;      // stop criterion met?
uchar            tf_group_crit;     // stop criterion met for all groups?
uchar            tf_event_crit;     // stop criterion met for all events?
uchar            tf_example_crit;   // stop criterion met for all examples?
stop_crit_t      tf_stop_func;      // stop evaluation function
uint             tf_stop_key;       // stop criterion packet key
uint             tf_stpn_key;       // stop network packet key

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

long_deriv_t   * t_output_deriv;    // derivative of the output value
delta_t        * t_deltas;

// history arrays
net_t          * t_net_history;
activation_t   * t_output_history;
activation_t   * t_target_history;
long_deriv_t   * t_output_deriv_history;
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
  uint stn_sent = 0;  // network_stop packets sent
  uint stn_recv = 0;  // network_stop packets received
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
  io_printf (IO_BUF, "threshold\n");

  // read the data specification header
  address_t data_address = data_specification_get_data_address ();
  if (!data_specification_read_header (data_address)) {
	  rt_error (RTE_SWERR);
  }

  // get addresses of all SDRAM regions
  // network configuration address
  address_t nt = data_specification_get_region (NETWORK, data_address);

  // initialize network configuration from SDRAM
  spin1_memcpy (&ncfg, nt, sizeof(network_conf_t));

  // core configuration address
  address_t dt = data_specification_get_region (CORE, data_address);

  // initialize core-specific configuration from SDRAM
  spin1_memcpy (&tcfg, dt, sizeof(t_conf_t));

  // inputs
  if (tcfg.input_grp)
  {
    it = (activation_t *) data_specification_get_region
		  (INPUTS, data_address);
  }

  // targets
  if (tcfg.output_grp)
  {
    tt = (activation_t *) data_specification_get_region
		  (TARGETS, data_address);
  }

  // example set
  es = (struct mlp_set *) data_specification_get_region
		  (EXAMPLE_SET, data_address);

  #ifdef DEBUG_CFG5
    io_printf (IO_BUF, "ne: %u\n", es->num_examples);
    io_printf (IO_BUF, "mt: %f\n", es->max_time);
    io_printf (IO_BUF, "nt: %f\n", es->min_time);
    io_printf (IO_BUF, "gt: %f\n", es->grace_time);
    io_printf (IO_BUF, "NaN: 0x%08x%\n", SPINN_FP_NaN);
  #endif

  // examples
  ex = (struct mlp_example *) data_specification_get_region
		  (EXAMPLES, data_address);

  #ifdef DEBUG_CFG5
    for (uint i = 0; i < es->num_examples; i++)
    {
      io_printf (IO_BUF, "nx[%u]: %u\n", i, ex[i].num);
      io_printf (IO_BUF, "nv[%u]: %u\n", i, ex[i].num_events);
      io_printf (IO_BUF, "vi[%u]: %u\n", i, ex[i].ev_idx);
      io_printf (IO_BUF, "xf[%u]: %f\n", i, ex[i].freq);
    }
  #endif

  // events
  ev = (struct mlp_event *) data_specification_get_region
		  (EVENTS, data_address);

  #ifdef DEBUG_CFG5
    uint evi = 0;
    for (uint i = 0; i < es->num_examples; i++)
    {
      for (uint j = 0; j < ex[i].num_events; j++)
      {
        io_printf (IO_BUF, "mt[%u][%u]: %f\n", i, j, ev[evi].max_time);
        io_printf (IO_BUF, "nt[%u][%u]: %f\n", i, j, ev[evi].min_time);
        io_printf (IO_BUF, "gt[%u][%u]: %f\n", i, j, ev[evi].grace_time);
        io_printf (IO_BUF, "ii[%u][%u]: %u\n", i, j, ev[evi].it_idx);
        evi++;
      }
    }
  #endif

  // routing keys
  rt = (uint *) data_specification_get_region
		  (ROUTING, data_address);

  #ifdef DEBUG_CFG0
    io_printf (IO_BUF, "og: %d\n", tcfg.output_grp);
    io_printf (IO_BUF, "ig: %d\n", tcfg.input_grp);
    io_printf (IO_BUF, "no: %d\n", tcfg.num_units);
    io_printf (IO_BUF, "fs: %d\n", tcfg.fwd_sync_expected);
    io_printf (IO_BUF, "bs: %d\n", tcfg.bkp_sync_expected);
    io_printf (IO_BUF, "wo: %d\n", tcfg.write_out);
    io_printf (IO_BUF, "wb: %d\n", tcfg.write_blk);
    io_printf (IO_BUF, "ie: %d\n", tcfg.out_integr_en);
    io_printf (IO_BUF, "dt: %f\n", tcfg.out_integr_dt);
    io_printf (IO_BUF, "np: %d\n", tcfg.num_out_procs);
    io_printf (IO_BUF, "pl: %d\n", tcfg.procs_list[0]);
    io_printf (IO_BUF, "pl: %d\n", tcfg.procs_list[1]);
    io_printf (IO_BUF, "pl: %d\n", tcfg.procs_list[2]);
    io_printf (IO_BUF, "pl: %d\n", tcfg.procs_list[3]);
    io_printf (IO_BUF, "pl: %d\n", tcfg.procs_list[4]);
    io_printf (IO_BUF, "wc: %f\n", tcfg.weak_clamp_strength);
    io_printf (IO_BUF, "io: %f\n", SPINN_LCONV_TO_PRINT(
    			tcfg.initOutput, SPINN_ACTIV_SHIFT));
    io_printf (IO_BUF, "gc: %k\n", tcfg.group_criterion);
    io_printf (IO_BUF, "cf: %d\n", tcfg.criterion_function);
    io_printf (IO_BUF, "fg: %d\n", tcfg.is_first_output_group);
    io_printf (IO_BUF, "lg: %d\n", tcfg.is_last_output_group);
    io_printf (IO_BUF, "ef: %d\n", tcfg.error_function);
    io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
    io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
    io_printf (IO_BUF, "sk: 0x%08x\n", rt[STP]);
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

  // allocate memory and initialize variables
  uint rcode = t_init ();

  return (rcode);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// check exit code and print details of the state
// ------------------------------------------------------------------------
void done (uint ec)
{
  // report problems -- if any
  switch (ec)
  {
    case SPINN_NO_ERROR:
      io_printf (IO_BUF, "simulation OK\n");

      break;

    case SPINN_QUEUE_FULL:
      io_printf (IO_BUF, "packet queue full\n");

      break;

    case SPINN_MEM_UNAVAIL:
      io_printf (IO_BUF, "malloc failed\n");

      break;

    case SPINN_UNXPD_PKT:
      io_printf (IO_BUF, "unexpected packet received - abort!\n");

      break;

    case SPINN_TIMEOUT_EXIT:
      io_printf (IO_BUF, "timeout (h:%u e:%u p:%u t:%u) - abort!\n",
                 epoch, example, phase, tick
                );

      #ifdef DEBUG_VRB
        io_printf (IO_BUF, "(tactive:%u ta:0x%08x/0x%08x tb:0x%08x/0x%08x)\n",
                    t_active, tf_arrived, tcfg.num_units,
                    tb_arrived, tcfg.num_units
                  );
        io_printf (IO_BUF, "(tsd:%u tsa:0x%08x/0x%08x)\n",
                    t_sync_done, t_sync_arrived, tcfg.fwd_sync_expected
                  );
      #endif

      if (tcfg.write_out)  // make sure the output monitor closes!
      {
        send_outputs_to_host (SPINN_HOST_FINAL, tick);
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
// timer callback: check that there has been progress in execution.
// If no progress has been made terminate with SPINN_TIMEOUT_EXIT exit code.
// ------------------------------------------------------------------------
void timeout (uint ticks, uint null)
{
  // check if progress has been made
  if ((to_epoch == epoch) && (to_example == example) && (to_tick == tick))
  {
    // exit and report timeout
    spin1_exit (SPINN_TIMEOUT_EXIT);
  }
  else
  {
    // update checked variables
    to_epoch   = epoch;
    to_example = example;
    to_tick    = tick;
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// main: register callbacks and initialize basic system variables
// ------------------------------------------------------------------------
void c_main ()
{
  // say hello,
  io_printf (IO_BUF, ">> mlp\n");

  // get this core's IDs,
  chipID = spin1_get_chip_id();
  coreID = spin1_get_core_id();

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
  spin1_callback_on (MCPL_PACKET_RECEIVED, t_receivePacket, SPINN_PACKET_P);

  // go,
  io_printf (IO_BUF, "-----------------------\n");
  io_printf (IO_BUF, "starting simulation\n");

  #ifdef PROFILE
    uint start_time = tc[T2_COUNT];
    io_printf (IO_BUF, "start count: %u\n", start_time);
  #endif

  // start execution and get exit code,
  exit_code = spin1_start (SYNC_WAIT);

  #ifdef PROFILE
    uint final_time = tc[T2_COUNT];
    io_printf (IO_BUF, "final count: %u\n", final_time);
    io_printf (IO_BUF, "execution time: %u us\n",
                  (start_time - final_time) / SPINN_TIMER2_DIV);
  #endif

  // report results,
  done (exit_code);

  io_printf (IO_BUF, "stopping simulation\n");
  io_printf (IO_BUF, "-----------------------\n");

  // and say goodbye
  io_printf (IO_BUF, "<< mlp\n");
}
// ------------------------------------------------------------------------
