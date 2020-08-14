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

#include "init_i.h"
#include "comms_i.h"
#include "process_i.h"


// ------------------------------------------------------------------------
// input core main routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// input core constants
// ------------------------------------------------------------------------
// list of procedures for the FORWARD phase in the input pipeline. The order is
// relevant, as the index is defined in mlp_params.h
in_proc_t const
  i_in_procs[SPINN_NUM_IN_PROCS] =
  {
    in_integr, in_soft_clamp
  };

// list of procedures for the BACKPROP phase. Order is relevant, as the index
// needs to be the same as in the FORWARD phase. In case a routine is not
// available, then a NULL should replace the call
in_proc_back_t const
  i_in_back_procs[SPINN_NUM_IN_PROCS] =
  {
    in_integr_back, NULL
  };

// list of procedures for the initialisation of the input pipeline. Order
// is relevant, as the index needs to be the same as in the FORWARD phase. In
// case one routine is not intended to be available because no initialisation
// is required, then a NULL should replace the call
in_proc_init_t const
  i_init_in_procs[SPINN_NUM_IN_PROCS] =
  {
      init_in_integr, NULL
  };
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
uint chipID;               // 16-bit (x, y) chip ID
uint coreID;               // 5-bit virtual core ID

uint fwdKey;               // packet ID for FORWARD-phase data
uint bkpKey;               // packet ID for BACKPROP-phase data

uint32_t stage_step;       // current stage step
uint32_t stage_num_steps;  // current stage number of steps

uchar        net_stop;     // network stop decision
uchar        net_stop_rdy; // ready to deal with network stop decision

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
uchar        tick_stop;    // current tick stop decision

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
uint             * rt;     // multicast routing keys data
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// network, core and stage configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t ncfg;           // network-wide configuration parameters
i_conf_t       icfg;           // input core configuration parameters
stage_conf_t   xcfg;           // stage configuration parameters
address_t      xadr;           // stage configuration SDRAM address
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// input core variables
// ------------------------------------------------------------------------
// input cores process the input values through a sequence of functions.
// ------------------------------------------------------------------------
long_net_t     * i_nets;            // unit nets computed in current tick
long_delta_t   * i_deltas;          // deltas computed in current tick
pkt_queue_t      i_pkt_queue;       // queue to hold received packets
uchar            i_active;          // processing packets from queue?

long_net_t     * i_last_integr_net; //last INTEGRATOR output value
long_delta_t   * i_last_integr_delta; //last INTEGRATOR delta value

uint             i_it_idx;          // index into current inputs/targets

// FORWARD phase specific
// (net processing)
scoreboard_t     if_done;           // current tick net computation done
uint             if_thrds_pend;     // thread semaphore

// BACKPROP phase specific
// (delta processing)
long_delta_t   * ib_init_delta;     // initial delta value for every tick
scoreboard_t     ib_done;           // current tick delta computation done

uint           * i_bkpKey;          // i cores have one bkpKey per partition

// history arrays
long_net_t     * i_net_history;   //sdram pointer where to store input history
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
uint stp_sent;  // stop packets sent
uint stp_recv;  // stop packets received
uint stn_recv;  // network_stop packets received
uint wrng_phs;  // packets received in wrong phase
uint wrng_pth;  // unexpected processing thread
uint wrng_cth;  // unexpected comms thread
uint wrng_sth;  // unexpected stop thread
uint tot_tick;  // total number of ticks executed
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
    // report timeout error
    stage_done (SPINN_TIMEOUT_EXIT, 0);
  }
  else
  {
    // update checked variables
    to_epoch   = epoch;
    to_example = example_cnt;
    to_tick    = tick;
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
void c_main ()
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

  // set up timer (used for background deadlock check),
  spin1_set_timer_tick (SPINN_TIMER_TICK_PERIOD);
  spin1_callback_on (TIMER_TICK, timeout, SPINN_TIMER_P);

  // set up packet received callbacks,
  spin1_callback_on (MC_PACKET_RECEIVED, i_receivePacket, SPINN_PACKET_P);
  spin1_callback_on (MCPL_PACKET_RECEIVED, i_receivePacket, SPINN_PACKET_P);

  // setup simulation,
  simulation_set_start_function (get_started);

  // and start execution
  simulation_run ();
}
// ------------------------------------------------------------------------
