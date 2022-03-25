/*
 * Copyright (c) 2015-2021 The University of Manchester
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

// front-end-common
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_w.h"
#include "comms_w.h"
#include "process_w.h"


// ------------------------------------------------------------------------
// weight core main routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// weight core constants
// ------------------------------------------------------------------------
// list of procedures for updating of weights. The order is relevant, as
// the indices are specified in mlp_params.h
weight_update_t const
  w_update_procs[SPINN_NUM_UPDATE_PROCS] =
  {
    steepest_update_weights, momentum_update_weights, dougsmomentum_update_weights
  };
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
uint chipID;               // 16-bit (x, y) chip ID
uint coreID;               // 5-bit virtual core ID

uint fwdKey;               // packet ID for FORWARD-phase data
uint bkpKey;               // packet ID for BACKPROP-phase data
uint ldsKey;               // packet ID for link delta summation
uint fsgKey;               // packet ID for FORWARD sync generation

uint32_t stage_step;       // current stage step
uint32_t stage_num_steps;  // current stage number of steps

uchar        epoch_rdy;    // this tick completed an epoch?
uchar        net_stop_rdy; // ready to deal with network stop decision

uchar        tick_stop;    // current tick stop decision
uchar        net_stop;     // network stop decision

uint         epoch;        // current training iteration
uint         example_cnt;  // example count in epoch
uint         example_inx;  // current example index
uint         evt;          // current event in example
uint         num_events;   // number of events in current example
uint         event_idx;    // index into current event
proc_phase_t phase;        // FORWARD or BACKPROP

uint         dlrv;         // deadlock recovery mode

uint         max_ticks;    // maximum number of ticks in current event
uint         min_ticks;    // minimum number of ticks in current event
uint         tick;         // current tick in phase
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// data structures in regions of SDRAM
// ------------------------------------------------------------------------
mlp_set_t        * es;     // example set data
mlp_example_t    * ex;     // example data
weight_t         * wt;     // initial connection weights
uint             * rt;     // multicast routing keys data
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// network, core and stage configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t ncfg;           // network-wide configuration parameters
w_conf_t       wcfg;           // weight core configuration parameters
stage_conf_t   xcfg;           // stage configuration parameters
address_t      xadr;           // stage configuration SDRAM address
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// weight core variables
// ------------------------------------------------------------------------
// weight cores compute net and error block dot-products (b-d-p),
// and weight updates.
// ------------------------------------------------------------------------
weight_t       * * w_weights;         // connection weights block
long_wchange_t * * w_wchanges;        // accumulated weight changes
activation_t     * w_outputs[2];      // unit outputs for b-d-p
long_delta_t   * * w_link_deltas;     // computed link deltas
error_t          * w_errors;          // computed errors next tick
pkt_queue_t        w_pkt_queue;       // queue to hold received packets
fpreal             w_delta_dt;        // scaling factor for link deltas
lds_t              w_lds_final;       // final link delta sum
scoreboard_t       w_sync_arrived;    // keep count of expected sync packets

// FORWARD phase specific variables
// (net b-d-p computation)
// Two sets of received unit outputs are kept:
// procs = in use for current b-d-p computation
// comms = being received for next tick
uint             wf_procs;          // pointer to processing unit outputs
uint             wf_comms;          // pointer to received unit outputs
scoreboard_t     wf_arrived;        // keep count of received unit outputs
uint             wf_thrds_pend;     // thread semaphore

// BACKPROP phase specific variables
// (error b-d-p computation)
uchar            wb_active;         // processing deltas from queue?
scoreboard_t     wb_arrived;        // keep count of received deltas
uint             wb_thrds_pend;     // thread semaphore
weight_update_t  wb_update_func;    // weight update function

// history arrays
activation_t   * w_output_history;  // history array for outputs
// ------------------------------------------------------------------------


#ifdef DEBUG
// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
uint sent_fwd;  // packets sent in FORWARD phase
uint sent_bkp;  // packets sent in BACKPROP phase
uint recv_fwd;  // packets received in FORWARD phase
uint recv_bkp;  // packets received in BACKPROP phase
uint pkt_fwbk;  // unused packets received in FORWARD phase
uint pkt_bwbk;  // unused packets received in BACKPROP phase
uint spk_recv;  // sync packets received
uint fsg_sent;  // forward sync generation packets sent (current tick)
uint stp_sent;  // stop packets sent
uint stp_recv;  // stop packets received
uint stn_recv;  // network_stop packets received
uint dlr_recv;  // deadlock recovery packets received
uint lds_sent;  // link_delta packets sent
uint lds_recv;  // link_delta packets received
uint wrng_fph;  // FORWARD packets received in wrong phase
uint wrng_bph;  // BACKPROP packets received in wrong phase
uint wght_ups;  // number of weight updates done
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

  // set up packet received callbacks,
  spin1_callback_on (MC_PACKET_RECEIVED, w_receiveControlPacket, SPINN_PACKET_P);
  spin1_callback_on (MCPL_PACKET_RECEIVED, w_receiveDataPacket, SPINN_PACKET_P);

  // set up user event callback to send some control packets
  spin1_callback_on (USER_EVENT, w_sendControlPacket, SPINN_W_SEND_P);

  // setup simulation,
  simulation_set_start_function (get_started);

  // and start execution
  simulation_run ();
}
// ------------------------------------------------------------------------
