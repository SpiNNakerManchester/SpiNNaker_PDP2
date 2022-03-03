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

#include "init_s.h"
#include "comms_s.h"


// ------------------------------------------------------------------------
// sum core main routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
uint chipID;               // 16-bit (x, y) chip ID
uint coreID;               // 5-bit virtual core ID

uint fwdKey;               // packet ID for FORWARD-phase data
uint bkpKey;               // packet ID for BACKPROP-phase data
uint ldsKey;               // packet ID for link delta summation
uint bpsKey;               // packet ID for BACKPROP synchronisation
uint fsgKey;               // packet ID for FORWARD sync generation

uint32_t stage_step;       // current stage step
uint32_t stage_num_steps;  // current stage number of steps

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
uint         max_ticks;    // maximum number of ticks in current event
uint         min_ticks;    // minimum number of ticks in current event
uint         tick;         // current tick in phase
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// data structures in regions of SDRAM
// ------------------------------------------------------------------------
mlp_set_t        * es;     // example set data
mlp_example_t    * ex;     // example data
uint             * rt;     // multicast routing keys data
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// network, core and stage configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t ncfg;           // network-wide configuration parameters
s_conf_t       scfg;           // sum core configuration parameters
stage_conf_t   xcfg;           // stage configuration parameters
address_t      xadr;           // stage configuration SDRAM address
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// sum core variables
// ------------------------------------------------------------------------
// sum cores compute unit nets and errors (accumulate b-d-ps).
// ------------------------------------------------------------------------
long_net_t     * s_nets;            // unit nets computed in current tick
long_error_t   * s_errors;          // errors computed in current tick
pkt_queue_t      s_pkt_queue;       // queue to hold received packets
uchar            s_active;          // processing packets from queue?
lds_t            s_lds_part;        // partial link delta sum

// FORWARD phase specific
// (net computation)
scoreboard_t   * sf_arrived;        // keep count of expected net b-d-p
scoreboard_t     sf_done;           // current tick net computation done
uint             sf_thrds_pend;     // thread semaphore
uint             sf_thrds_init;     // thread semaphore initialisation

// BACKPROP phase specific
// (error computation)
scoreboard_t   * sb_arrived;        // keep count of expected error b-d-p
scoreboard_t     sb_done;           // current tick error computation done
uint             sb_thrds_pend;     // thread semaphore
uint             sb_thrds_init;     // thread semaphore initialisation
scoreboard_t     s_lds_arrived;     // keep count of the number of partial link delta sums
scoreboard_t     s_sync_arrived;    // keep count of expected sync packets
scoreboard_t     s_fsgn_arrived;    // keep count of forward sync gen packets
uint             s_fsgn_expected;   // expected count of forward sync gen packets
// ------------------------------------------------------------------------


#ifdef DEBUG
// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
uint pkt_sent;  // total packets sent
uint sent_fwd;  // packets sent in FORWARD phase
uint sent_bkp;  // packets sent in BACKPROP phase
uint pkt_recv;  // total packets received
uint recv_fwd;  // packets received in FORWARD phase
uint recv_bkp;  // packets received in BACKPROP phase
uint spk_recv;  // sync packets received
uint fsg_sent;  // forward sync generation packets sent
uint fsg_recv;  // forward sync generation packets received
uint bsg_sent;  // BACKPROP sync generation packets sent
uint bsg_recv;  // BACKPROP sync generation packets received
uint stp_sent;  // stop packets sent
uint stp_recv;  // stop packets received
uint stn_recv;  // network_stop packets received
uint dlr_recv;  // deadlock recovery packets received
uint lds_sent;  // link_delta packets sent
uint lds_recv;  // link_delta packets received
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
  var_init (TRUE);

  // set up packet received callbacks,
  spin1_callback_on (MC_PACKET_RECEIVED, s_receiveControlPacket, SPINN_PACKET_P);
  spin1_callback_on (MCPL_PACKET_RECEIVED, s_receiveDataPacket, SPINN_PACKET_P);

  // setup simulation,
  simulation_set_start_function (get_started);

  // and start execution
  simulation_run ();
}
// ------------------------------------------------------------------------
