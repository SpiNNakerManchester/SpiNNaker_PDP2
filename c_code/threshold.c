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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// SpiNNaker API
#include "spin1_api.h"

// front-end-common
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"
#include "mlp_externs.h"

#include "init_t.h"
#include "comms_t.h"
#include "process_t.h"


// ------------------------------------------------------------------------
// threshold core main routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// threshold core constants
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

// list of procedures for the initialisation of the output pipeline. The order
// is relevant, as the index needs to be the same as in the FORWARD phase. In
// case one routine is not intended to be available because no initialisation
// is required, then a NULL should replace the call
out_proc_init_t const
  t_init_out_procs[SPINN_NUM_OUT_PROCS] =
  {
      NULL, init_out_integr, init_out_hard_clamp, init_out_weak_clamp, NULL
  };

// list of procedures for the evaluation of the convergence (and stopping)
// criteria. The order is relevant, as the indices are specified in mlp_params.h
// A NULL routine does not evaluate any convergence criterion and therefore
// computation will continue for the defined maximum number of ticks
stop_crit_t const
  t_stop_procs[SPINN_NUM_STOP_PROCS] =
  {
    NULL, std_stop_crit, max_stop_crit
  };

// list of procedures for the evaluation of the errors between the output and
// the target values of the output groups. The order is relevant, as the indices
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

uint fwdKey;               // packet ID for FORWARD-phase data
uint bkpKey;               // packet ID for BACKPROP-phase data
uint bpsKey;               // packet ID for BACKPROP synchronisation

uint32_t stage_step;       // current stage step
uint32_t stage_num_steps;  // current stage number of steps
uint32_t stage_rec_flags;  // current stage recording flags

uchar        net_stop_rdy; // ready to deal with network stop decision

uchar        tick_stop;    // current tick stop decision
uchar        net_stop;     // network stop decision

uint         epoch;        // current training iteration
uint         example_cnt;  // example count in epoch
uint         train_cnt;    // latest example trained
uint         test_cnt;     // latest example tested
uint         example_inx;  // current example index
uint         evt;          // current event in example
uint         max_evt;      // the last event reached in the current example
uint         num_events;   // number of events in current example
uint         event_idx;    // index into current event
proc_phase_t phase;        // FORWARD or BACKPROP

uint         dlrv;         // deadlock recovery mode

uint         max_ticks;    // maximum number of ticks in current event
uint         min_ticks;    // minimum number of ticks in current event
uint         tick;         // current tick in phase
uint         ev_tick;      // current tick in event

uint         to_epoch   = 0;
uint         to_example = 0;
uint         to_tick    = 0;
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// data structures in regions of SDRAM
// ------------------------------------------------------------------------
mlp_set_t        * es;     // example set data
mlp_example_t    * ex;     // example data
mlp_event_t      * ev;     // event data
activation_t     * it;     // example inputs
activation_t     * tt;     // example targets
uint             * rt;     // multicast routing keys data
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// network, core and stage configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t ncfg;           // network-wide configuration parameters
t_conf_t       tcfg;           // threshold core configuration parameters
stage_conf_t   xcfg;           // stage configuration parameters
address_t      xadr;           // stage configuration SDRAM address
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// threshold core variables
// ------------------------------------------------------------------------
// threshold cores compute unit outputs and error deltas.
// ------------------------------------------------------------------------
activation_t   * t_outputs;         // current tick unit outputs
net_t          * t_nets;            // nets received from input cores
error_t        * t_errors[2];       // error banks: current and next tick
activation_t   * t_last_integr_output;  //last INTEGRATOR output value
long_deriv_t   * t_last_integr_output_deriv; //last INTEGRATOR output deriv value
activation_t   * t_last_integr_output_dlrv;       // deadlock recovery
long_deriv_t   * t_last_integr_output_deriv_dlrv; // deadlock recovery
activation_t   * t_instant_outputs; // current output value stored for the backward pass
short_activ_t  * t_out_hard_clamp_data; //values injected by hard clamps
short_activ_t  * t_out_weak_clamp_data; //values injected by weak clamps
uint             t_it_idx;          // index into current inputs/targets
pkt_queue_t      t_pkt_queue;       // queue to hold received nets
uint             t_dlrv_cnt;        // count deadlock recovery attempts
uint             t_dlrv_rep;        // count repeated deadlock recovery attempts

// FORWARD phase specific
// (output computation)
uchar            tf_active;         // processing FWD-phase packet queue?
scoreboard_t     tf_arrived;        // keep count of expected nets
uint             tf_thrds_pend;     // thread semaphore
uint             tf_thrds_init;     // thread semaphore initial value
uchar            tf_crit_prev;      // criterion value received
scoreboard_t     tf_crit_arrived;   // keep count of expected crit pkts
uchar            tf_stop_crit;      // stop criterion met?
uchar            tf_group_crit;     // stop criterion met for all groups?
uchar            tf_event_crit;     // stop criterion met for all events?
uchar            tf_example_crit;   // stop criterion met for all examples?
error_t          t_group_criterion; // convergence criterion value
test_results_t   t_test_results;    // test results to report to host
stop_crit_t      tf_stop_func;      // stop evaluation function
uint             tf_stop_key;       // stop criterion packet key
uint             tf_stpn_key;       // stop network packet key
uint             tf_dlrv_key;       // deadlock recovery packet key

// BACKPROP phase specific
// (error delta computation)
uint             tb_procs;          // pointer to processing errors
uint             tb_comms;          // pointer to received errors
scoreboard_t     tb_arrived;        // keep count of expected errors
uint             tb_thrds_pend;     // thread semaphore
uint             tb_thrds_init;     // thread semaphore initial value
scoreboard_t     tb_bsgn_arrived;   // keep count of backprop sync gen pkts
scoreboard_t     tb_bsgn_expected;  // keep count of backprop sync gen pkts

int              t_max_output_unit; // unit with highest output
int              t_max_target_unit; // unit with highest target
activation_t     t_max_output;      // highest output value
activation_t     t_max_target;      // highest target value

long_deriv_t   * t_output_deriv;    // derivative of the output value
delta_t        * t_deltas;

// recording control
uchar            t_rec_outputs;     // record outputs to SDRAM
uchar            t_rec_results;     // record test results to SDRAM
uchar            t_rec_tick_data;   // record tick data to SDRAM
uchar            t_rec_step_updt;   // update recording step

// history arrays
net_t          * t_net_history;
activation_t   * t_output_history;
long_deriv_t   * t_output_deriv_history;
// ------------------------------------------------------------------------


#ifdef DEBUG
// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
uint sent_fwd;  // packets sent in FORWARD phase
uint sent_bkp;  // packets sent in BACKPROP phase
uint recv_fwd;  // packets received in FORWARD phase
uint recv_bkp;  // packets received in BACKPROP phase
uint spk_sent;  // sync packets sent
uint spk_recv;  // sync packets received
uint crt_sent;  // criterion packets sent (current tick)
uint crt_recv;  // criterion packets received (current tick)
uint fsg_recv;  // forward sync generation packets received (current tick)
uint bsg_sent;  // BACKPROP sync generation packets sent (current tick)
uint bsg_recv;  // BACKPROP sync generation packets received (current tick)
uint stp_sent;  // stop packets sent
uint stp_recv;  // stop packets received
uint stn_sent;  // network_stop packets sent
uint stn_recv;  // network_stop packets received
uint dlr_sent;  // deadlock recovery packets sent
uint dlr_recv;  // deadlock recovery packets received
uint wrng_fph;  // FORWARD packets received in wrong phase
uint wrng_bph;  // BACKPROP packets received in wrong phase
uint wrng_pth;  // unexpected processing thread
uint wrng_cth;  // unexpected comms thread
uint wrng_sth;  // unexpected stop thread

uint tot_tick;  // total number of ticks executed
// ------------------------------------------------------------------------
#endif


#ifdef PROFILE
// ------------------------------------------------------------------------
// PROFILER variables
// ------------------------------------------------------------------------
uint prf_fwd_min;  // minimum FORWARD processing time
uint prf_fwd_max;  // maximum FORWARD processing time
uint prf_bkp_min;  // minimum BACKPROP processing time
uint prf_bkp_max;  // maximum BACKPROP processing time
// ------------------------------------------------------------------------
#endif


// ------------------------------------------------------------------------
// timer callback: check that there has been progress in execution.
// If no progress has been made terminate with SPINN_TIMEOUT_EXIT code.
// ------------------------------------------------------------------------
void timeout (uint ticks, uint unused)
{
  (void) ticks;
  (void) unused;

  // check if progress has been made
  if ((to_epoch == epoch) && (to_example == example_cnt) && (to_tick == tick))
  {
#ifdef DEBUG
    dlr_sent++;
    if (t_dlrv_cnt > 0) t_dlrv_rep++;
#endif

    t_dlrv_cnt++;
    if (t_dlrv_cnt >= SPINN_DLRV_MAX_CNT)
    {
      // send abort on deadlock packet to all other cores,
      while (!spin1_send_mc_packet(tf_dlrv_key | SPINN_DLRV_ABRT, 0, NO_PAYLOAD));

      // and report timeout error
      stage_done (SPINN_TIMEOUT_EXIT, 0);
    }
    else
    {
      // send deadlock recovery packet to all other cores,
      while (!spin1_send_mc_packet(tf_dlrv_key, 0, NO_PAYLOAD));

      // and restart tick
      //NOTE: this core sends but does *not* receive dlrv packets!
      t_dlrv_packet ();
    }
  }
  else
  {
    // update checked variables,
    to_epoch   = epoch;
    to_example = example_cnt;
    to_tick    = tick;

    // and initialise deadlock recovery attempt count
    t_dlrv_cnt = 0;

  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// kick start simulation
//NOTE: workaround for an FEC bug
// ------------------------------------------------------------------------
void get_started (void)
{
  // start timer,
  vic[VIC_ENABLE] = (1 << TIMER1_INT);
  tc[T1_CONTROL] = 0xe2;

  // redefine start function,
  simulation_set_start_function (stage_start);

  // and run new start function
  stage_start ();
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// main: register callbacks and initialise basic system variables
// ------------------------------------------------------------------------
void c_main (void)
{
  // get core IDs,
  chipID = spin1_get_chip_id ();
  coreID = spin1_get_core_id ();

  // initialise configurations from SDRAM,
  uint exit_code = cfg_init ();
  if (exit_code != SPINN_NO_ERROR)
  {
    // report results and abort
    stage_done (exit_code, 0);
  }

  // allocate memory in DTCM and SDRAM,
  exit_code = mem_init ();
  if (exit_code != SPINN_NO_ERROR)
  {
    // report results and abort
    stage_done (exit_code, 0);
  }

  // initialise variables,
  var_init (TRUE, TRUE);

  // set up timer (used for background deadlock check),
  if (tcfg.is_last_output)
  {
    spin1_set_timer_tick (SPINN_TIMER_TICK_PERIOD);
    spin1_callback_on (TIMER_TICK, timeout, SPINN_TIMER_P);
  }

  // set up packet received callbacks,
  spin1_callback_on (MC_PACKET_RECEIVED, t_receiveControlPacket, SPINN_PACKET_P);
  spin1_callback_on (MCPL_PACKET_RECEIVED, t_receiveDataPacket, SPINN_PACKET_P);

  // set up user event callback to send some control packets
  spin1_callback_on (USER_EVENT, t_sendControlPacket, SPINN_T_SEND_P);

  // setup simulation,
  simulation_set_start_function (get_started);

  // and start execution
  simulation_run ();
}
// ------------------------------------------------------------------------
