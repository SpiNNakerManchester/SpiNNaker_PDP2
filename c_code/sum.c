// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"  // allows compiler to check extern types!

#include "init_s.h"
#include "comms_s.h"

// main methods for the sum core

// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
uint chipID;               // 16-bit (x, y) chip ID
uint coreID;               // 5-bit virtual core ID

uint fwdKey;               // 32-bit packet ID for FORWARD phase
uint bkpKey;               // 32-bit packet ID for BACKPROP phase
uint ldstKey;              // 32-bit packet ID for link delta summation totals
uint ldsrKey;              // 32-bit packet ID for link delta summation reports

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
uchar        tick_stop;    // current tick stop decision

uint         to_epoch   = 0;
uint         to_example = 0;
uint         to_tick    = 0;

// ------------------------------------------------------------------------
// data structures in regions of SDRAM
// ------------------------------------------------------------------------
mlp_example_t    *ex; // example data
uint             *rt; // multicast routing keys data

// ------------------------------------------------------------------------
// network and core configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t ncfg;           // network-wide configuration parameters
s_conf_t       scfg;           // sum core configuration parameters
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// sum core variables
// ------------------------------------------------------------------------
// sum cores compute unit nets and errors (acummulate b-d-ps).
// ------------------------------------------------------------------------
long_net_t     * s_nets[2];         // unit nets computed in current tick
long_error_t   * s_errors[2];       // errors computed in current tick
pkt_queue_t      s_pkt_queue;       // queue to hold received b-d-ps
uchar            s_active;          // processing b-d-ps from queue?
lds_t            s_lds_part;        // partial link delta sum

// FORWARD phase specific
// (net computation)
scoreboard_t   * sf_arrived[2];     // keep track of expected net b-d-p
scoreboard_t     sf_done;           // current tick net computation done
uint             sf_thrds_pend;     // sync. semaphore: proc & stop

// BACKPROP phase specific
// (error computation)
scoreboard_t   * sb_arrived[2];     // keep track of expected error b-d-p
scoreboard_t     sb_done;           // current tick error computation done
uint             sb_thrds_pend;     // sync. semaphore: proc & stop
scoreboard_t     s_ldsa_arrived;    // keep track of the number of partial link delta sums
scoreboard_t     s_ldst_arrived;    // keep track of the number of link delta sum totals
// ------------------------------------------------------------------------

#ifdef DEBUG
// ------------------------------------------------------------------------
// DEBUG variables
// ------------------------------------------------------------------------
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
uint stn_recv = 0;  // network_stop packets received
uint lda_recv = 0;  // partial link_delta packets received
uint ldt_sent = 0;  // total link_delta packets sent
uint ldt_recv = 0;  // total link_delta packets received
uint ldr_sent = 0;  // link_delta packets sent
uint wrng_phs = 0;  // packets received in wrong phase
uint wrng_tck = 0;  // FORWARD packets received in wrong tick
uint wrng_btk = 0;  // BACKPROP packets received in wrong tick
uint wght_ups = 0;  // number of weight updates done
uint tot_tick = 0;  // total number of ticks executed
// ------------------------------------------------------------------------
#endif


// ------------------------------------------------------------------------
// load configuration from SDRAM and initialise variables
// ------------------------------------------------------------------------
uint init ()
{
  io_printf (IO_BUF, "sum\n");

  // read the data specification header
  data_specification_metadata_t * data =
          data_specification_get_data_address();
  if (!data_specification_read_header (data))
  {
	  return (SPINN_CFG_UNAVAIL);
  }

  // set up the simulation interface (system region)
  //NOTE: these variables are not used!
  uint32_t n_steps, run_forever, step;
  if (!simulation_steps_initialise(
      data_specification_get_region(SYSTEM, data),
      APPLICATION_NAME_HASH, &n_steps, &run_forever, &step, 0, 0))
  {
    return (SPINN_CFG_UNAVAIL);
  }

  // network configuration address
  address_t nt = data_specification_get_region (NETWORK, data);

  // initialise network configuration from SDRAM
  spin1_memcpy (&ncfg, nt, sizeof (network_conf_t));

  // core configuration address
  address_t dt = data_specification_get_region (CORE, data);

  // initialise core-specific configuration from SDRAM
  spin1_memcpy (&scfg, dt, sizeof (s_conf_t));

  // examples
  ex = (mlp_example_t *) data_specification_get_region
		  (EXAMPLES, data);

  // routing keys
  rt = (uint *) data_specification_get_region
		  (ROUTING, data);

#ifdef DEBUG_CFG0
  io_printf (IO_BUF, "nu: %d\n", scfg.num_units);
  io_printf (IO_BUF, "fe: %d\n", scfg.fwd_expected);
  io_printf (IO_BUF, "be: %d\n", scfg.bkp_expected);
  io_printf (IO_BUF, "ae: %d\n", scfg.ldsa_expected);
  io_printf (IO_BUF, "te: %d\n", scfg.ldst_expected);
  io_printf (IO_BUF, "uf: %d\n", scfg.update_function);
  io_printf (IO_BUF, "fg: %d\n", scfg.is_first_group);
  io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
  io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
  io_printf (IO_BUF, "lk: 0x%08x\n", rt[LDS]);
#endif

  // initialise epoch, example and event counters
  //TODO: alternative algorithms for choosing example order!
  epoch   = 0;
  example = 0;
  evt     = 0;

  // initialise phase
  phase = SPINN_FORWARD;

  // initialise number of events and event index
  num_events = ex[example].num_events;
  event_idx  = ex[example].ev_idx;

  // allocate memory and initialise variables
  uint rcode = s_init ();

  return (rcode);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// timer callback: check that there has been progress in execution.
// If no progress has been made terminate with SPINN_TIMEOUT_EXIT exit code.
// ------------------------------------------------------------------------
void timeout (uint ticks, uint unused)
{
  (void) ticks;
  (void) unused;

  // check if progress has been made
  if ((to_epoch == epoch) && (to_example == example) && (to_tick == tick))
  {
    // report timeout error
    stage_done (SPINN_TIMEOUT_EXIT);
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
  // say hello,
  io_printf (IO_BUF, ">> mlp\n");

  // get this core's IDs,
  chipID = spin1_get_chip_id();
  coreID = spin1_get_core_id();

  // initialise application,
  uint exit_code = init ();
  if (exit_code != SPINN_NO_ERROR)
  {
    // if init failed report results and abort simulation
    stage_done (exit_code);
  }

  // set up timer1 (used for background deadlock check),
  spin1_set_timer_tick (SPINN_TIMER_TICK_PERIOD);
  spin1_callback_on (TIMER_TICK, timeout, SPINN_TIMER_P);

  // set up packet received callbacks,
  spin1_callback_on (MC_PACKET_RECEIVED, s_receivePacket, SPINN_PACKET_P);
  spin1_callback_on (MCPL_PACKET_RECEIVED, s_receivePacket, SPINN_PACKET_P);

  // setup simulation,
  simulation_set_start_function (get_started);

  // start execution,
  simulation_run ();

  // and say goodbye
  io_printf (IO_BUF, "<< mlp\n");
}
// ------------------------------------------------------------------------
