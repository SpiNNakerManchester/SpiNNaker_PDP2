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

#ifndef __MLP_EXTERNS_H__
#define __MLP_EXTERNS_H__

// front-end-common types
#include "common-typedefs.h"


// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern uint coreID;               // 5-bit virtual core ID

extern uint fwdKey;               // packet ID for FORWARD-phase data
extern uint bkpKey;               // packet ID for BACKPROP-phase data
extern uint ldsKey;               // packet ID for link delta summation
extern uint bpsKey;               // packet ID for BACKPROP synchronisation
extern uint fsgKey;               // packet ID for FORWARD sync generation

extern uint32_t stage_step;       // current stage step
extern uint32_t stage_num_steps;  // current stage number of steps
extern uint32_t stage_rec_flags;  // current stage recording flags

extern uchar        epoch_rdy;    // this tick completed an epoch?
extern uchar        net_stop_rdy; // ready to deal with network stop decision

extern uchar        tick_stop;    // current tick stop decision
extern uchar        net_stop;     // network stop decision

extern uint         epoch;        // current training/testing iteration
extern uint         example_cnt;  // example count in epoch
extern uint         example_inx;  // current example index
extern uint         evt;          // current event in example
extern uint         max_evt;      // the last event reached in the current example
extern uint         num_events;   // number of events in current example
extern uint         event_idx;    // index into current event
extern uint         max_ticks;    // maximum number of ticks in current event
extern uint         min_ticks;    // minimum number of ticks in current event
extern uint         tick;         // current tick in phase
extern uint         ev_tick;      // current tick in event
extern proc_phase_t phase;        // FORWARD or BACKPROP

extern uint         dlrv;         // deadlock recovery mode

extern uint                 *rt; // multicast routing keys data
extern weight_t             *wt; // initial connection weights
extern struct mlp_set       *es; // example set data
extern struct mlp_example   *ex; // example data
extern struct mlp_event     *ev; // event data
extern activation_t         *it; // example inputs
extern activation_t         *tt; // example targets

// ------------------------------------------------------------------------
// network and core configurations
// ------------------------------------------------------------------------
extern network_conf_t ncfg;       // network-wide configuration parameters
extern w_conf_t       wcfg;       // weight core configuration parameters
extern s_conf_t       scfg;       // sum core configuration parameters
extern i_conf_t       icfg;       // input core configuration parameters
extern t_conf_t       tcfg;       // threshold core configuration parameters
extern stage_conf_t   xcfg;       // stage configuration parameters
extern address_t      xadr;       // stage configuration SDRAM address
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// weight core variables
// ------------------------------------------------------------------------
// global "constants"
// list of weight update procedures
extern weight_update_t const w_update_procs[SPINN_NUM_UPDATE_PROCS];

extern weight_t       * * w_weights;     // connection weights block
extern long_wchange_t * * w_wchanges;    // accumulated weight changes
extern activation_t     * w_outputs[2];  // unit outputs for b-d-p
extern long_delta_t   * * w_link_deltas; // computed link deltas
extern error_t          * w_errors;      // computed errors next tick
extern pkt_queue_t        w_pkt_queue;   // queue to hold received packets
extern fpreal             w_delta_dt;    // scaling factor for link deltas
extern lds_t              w_lds_final;   // final link delta sum
extern scoreboard_t       w_sync_arrived; // keep count of expected sync packets
extern uint               wf_procs;      // pointer to processing unit outputs
extern uint               wf_comms;      // pointer to received unit outputs
extern scoreboard_t       wf_arrived;    // keep count of received unit outputs
extern uint               wf_thrds_pend; // thread semaphore
extern uchar              wb_active;     // processing BKP-phase packet queue?
extern scoreboard_t       wb_arrived;    // keep count of received deltas
extern uint               wb_thrds_pend; // thread semaphore
extern weight_update_t    wb_update_func; // weight update function

// history arrays
extern activation_t     * w_output_history;
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// sum core variables
// ------------------------------------------------------------------------
extern long_net_t     * s_nets;         // unit nets computed in current tick
extern long_error_t   * s_errors;       // errors computed in current tick
extern pkt_queue_t      s_pkt_queue;    // queue to hold received packets
extern uchar            s_active;       // processing packets from queue?
extern lds_t            s_lds_part;     // partial link delta sum
extern scoreboard_t   * sf_arrived;     // keep count of expected net b-d-p
extern scoreboard_t     s_fsgn_arrived; // keep count of expected sync packets
extern uint             sf_thrds_pend;  // thread semaphore
extern uint             sf_thrds_init;  // thread semaphore initialisation
extern scoreboard_t   * sb_arrived;     // keep count of expected error b-d-p
extern scoreboard_t     sb_done;        // current tick error computation done
extern uint             sb_thrds_pend;  // thread semaphore
extern uint             sb_thrds_init;  // thread semaphore initialisation
extern scoreboard_t     s_lds_arrived;  // keep count of received link delta sums
extern scoreboard_t     s_bsgn_arrived; // keep count of expected sync packets
extern scoreboard_t     s_fsgn_arrived; // keep count of forward sync gen packets
extern uint             s_fsgn_expected;// expected count of forward sync gen packets
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// input core variables
// ------------------------------------------------------------------------
// global "constants"
//list of input pipeline procedures
extern in_proc_t      const i_in_procs[SPINN_NUM_IN_PROCS];
extern in_proc_back_t const i_in_back_procs[SPINN_NUM_IN_PROCS];
//list of initialisation procedures for input pipeline
extern in_proc_init_t const i_init_in_procs[SPINN_NUM_IN_PROCS];

extern long_net_t     * i_nets;        // unit nets computed in current tick
extern long_delta_t   * i_deltas;      // deltas computed in current tick
extern pkt_queue_t      i_pkt_queue;   // queue to hold received packets
extern uchar            i_active;      // processing packets from queue?
extern uint             i_it_idx;      // index into current inputs/targets
extern uint             if_thrds_pend; // thread semaphore
extern long_delta_t   * ib_init_delta; // initial delta value for every tick
extern uint             ib_thrds_pend; // thread semaphore
extern long_net_t     * i_last_integr_net;   // last INTEGRATOR output value
extern long_delta_t   * i_last_integr_delta; // last INTEGRATOR delta value
extern long_net_t     * i_last_integr_net_dlrv;   // deadlock recovery
extern long_delta_t   * i_last_integr_delta_dlrv; // deadlock recovery

// history arrays
extern long_net_t     * i_net_history; //sdram pointer where to store input history
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// threshold core variables
// ------------------------------------------------------------------------
// global "constants"
// list of output pipeline procedures
extern out_proc_t      const t_out_procs[SPINN_NUM_OUT_PROCS];
extern out_proc_back_t const t_out_back_procs[SPINN_NUM_OUT_PROCS];
// list of stop eval procedures
extern stop_crit_t     const t_stop_procs[SPINN_NUM_STOP_PROCS];
// list of initialisation procedures for output pipeline
extern out_proc_init_t const t_init_out_procs[SPINN_NUM_OUT_PROCS];
extern out_error_t     const t_out_error[SPINN_NUM_ERROR_PROCS];

extern activation_t   * t_outputs;     // current tick unit outputs
extern net_t          * t_nets;        // nets received from input cores
extern error_t        * t_errors[2];   // error banks: current and next tick
extern activation_t   * t_last_integr_output;        // last INTEGRATOR output value
extern long_deriv_t   * t_last_integr_output_deriv; // last INTEGRATOR output deriv
extern activation_t   * t_last_integr_output_dlrv;       // deadlock recovery
extern long_deriv_t   * t_last_integr_output_deriv_dlrv; // deadlock recovery
extern activation_t   * t_instant_outputs; // output stored BACKPROP
extern uint             t_it_idx;      // index into current inputs/targets
extern pkt_queue_t      t_pkt_queue;   // queue to hold received packets
extern uint             t_dlrv_cnt;        // count deadlock recovery attempts
extern uint             t_dlrv_rep;        // count repeated deadlock recovery attempts
extern uchar            tf_active;     // processing FWD-phase packet queue?
extern scoreboard_t     tf_arrived;    // keep count of expected nets
extern uint             tf_thrds_pend; // thread semaphore
extern uint             tf_thrds_init;     // thread semaphore initial value
extern uchar            tf_crit_prev;  // criterion value received
extern scoreboard_t     tf_crit_arrived;  // keep count of expected crit pkts
extern uchar            tf_stop_crit;  // stop criterion met?
extern uchar            tf_group_crit;     // stop criterion met for all groups?
extern uchar            tf_event_crit;     // stop criterion met for all events?
extern uchar            tf_example_crit;   // stop criterion met for all examples?
extern error_t          t_group_criterion; // convergence criterion value
extern test_results_t   t_test_results;    // test results to report to host
extern stop_crit_t      tf_stop_func;  // stop evaluation function
extern uint             tf_stop_key;   // stop criterion packet key
extern uint             tf_stpn_key;   // stop network packet key
extern uint             tf_dlrv_key;   // deadlock recovery packet key
extern uint             tb_procs;      // pointer to processing errors
extern uint             tb_comms;      // pointer to received errors
extern scoreboard_t     tb_arrived;    // keep count of expected errors
extern uint             tb_thrds_pend; // thread semaphore
extern uint             tb_thrds_init; // thread semaphore initial value
extern scoreboard_t     tb_bsgn_arrived;   // keep count of backprop sync gen pkts
extern scoreboard_t     tb_bsgn_expected;  // keep count of backprop sync gen pkts
extern int              t_max_output_unit; // unit with highest output
extern int              t_max_target_unit; // unit with highest target
extern activation_t     t_max_output;      // highest output value
extern activation_t     t_max_target;      // highest target value
extern long_deriv_t   * t_output_deriv;
extern delta_t        * t_deltas;

// recording control
extern uchar            t_rec_outputs;    // record outputs to SDRAM
extern uchar            t_rec_results;    // record test results to SDRAM
extern uchar            t_rec_tick_data;  // record tick data to SDRAM
extern uchar            t_rec_step_updt;  // update recording step

// history arrays
extern net_t          * t_net_history;
extern activation_t   * t_output_history;
extern long_deriv_t   * t_output_deriv_history;
// ------------------------------------------------------------------------

#ifdef DEBUG
// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
extern uint sent_fwd;  // packets sent in FORWARD phase
extern uint sent_bkp;  // packets sent in BACKPROP phase
extern uint recv_fwd;  // packets received in FORWARD phase
extern uint recv_bkp;  // packets received in BACKPROP phase
extern uint pkt_fwbk;  // unused packets received in FORWARD phase
extern uint pkt_bwbk;  // unused packets received in BACKPROP phase
extern uint spk_sent;  // sync packets sent
extern uint spk_recv;  // sync packets received
extern uint crt_sent;  // criterion packets sent (current tick)
extern uint crt_recv;  // criterion packets received (current tick)
extern uint fsg_sent;  // FORWARD sync generation packets sent (current tick)
extern uint fsg_recv;  // FORWARD sync generation packets received (current tick)
extern uint bsg_sent;  // BACKPROP sync generation packets sent (current tick)
extern uint bsg_recv;  // BACKPROP sync generation packets received (current tick)
extern uint stp_sent;  // stop packets sent
extern uint stp_recv;  // stop packets received
extern uint stn_sent;  // network_stop packets sent
extern uint stn_recv;  // network_stop packets received
extern uint dlr_sent;  // deadlock recovery packets sent
extern uint dlr_recv;  // deadlock recovery packets received
extern uint lds_sent;  // link_delta packets sent
extern uint lds_recv;  // link_delta packets received
extern uint wght_ups;  // number of weight updates done
extern uint wrng_fph;  // FORWARD packets received in wrong phase
extern uint wrng_bph;  // BACKPROP packets received in wrong phase
extern uint wrng_pth;  // unexpected processing thread
extern uint wrng_cth;  // unexpected comms thread
extern uint wrng_sth;  // unexpected stop thread

extern uint tot_tick;  // total number of ticks executed
// ------------------------------------------------------------------------
#endif


#ifdef PROFILE
// ------------------------------------------------------------------------
// PROFILER variables
// ------------------------------------------------------------------------
extern uint prf_fwd_min;  // minimum FORWARD processing time
extern uint prf_fwd_max;  // maximum FORWARD processing time
extern uint prf_bkp_min;  // minimum BACKPROP processing time
extern uint prf_bkp_max;  // maximum BACKPROP processing time
// ------------------------------------------------------------------------
#endif


#endif
