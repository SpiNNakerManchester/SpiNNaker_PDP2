// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include <data_specification.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"  // allows compiler to check extern types!

#include "init_w.h"
#include "comms_w.h"
#include "process_w.h"

// main methods for the weight core

// ------------------------------------------------------------------------
// global "constants"
// ------------------------------------------------------------------------

// list of procedures for updating of weights. The order is relevant, as
// the indexes are specified in mlp_params.h
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

uint fwdKey;               // 32-bit packet ID for FORWARD phase
uint bkpKey;               // 32-bit packet ID for BACKPROP phase
uint ldsaKey;              // 32-bit packet ID for link delta summation

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
weight_t         * wt;     // initial connection weights
mlp_example_t    * ex;     // example data
uint             * rt;     // multicast routing keys data

// ------------------------------------------------------------------------
// network and core configurations (DTCM)
// ------------------------------------------------------------------------
network_conf_t     ncfg;   // network-wide configuration parameters
w_conf_t           wcfg;   // weight core configuration parameters
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
pkt_queue_t        w_delta_pkt_q;     // queue to hold received deltas
fpreal             w_delta_dt;        // scaling factor for link deltas
lds_t              w_lds_final;       // final link delta sum

// FORWARD phase specific variables
// (net b-d-p computation)
// Two sets of received unit outputs are kept:
// procs = in use for current b-d-p computation
// comms = being received for next tick
uint             wf_procs;          // pointer to processing unit outputs
uint             wf_comms;          // pointer to receiving unit outputs
scoreboard_t     wf_arrived;        // keeps track of received unit outputs
uint             wf_thrds_pend;     // sync. semaphore: comms, proc & stop
uint             wf_sync_key;       // FORWARD processing can start

// BACKPROP phase specific variables
// (error b-d-p computation)
uchar            wb_active;         // processing deltas from queue?
scoreboard_t     wb_arrived;        // keeps track of received deltas
uint             wb_thrds_pend;     // sync. semaphore: comms, proc & stop
uint             wb_sync_key;       // BACKPROP processing can start
weight_update_t  wb_update_func;    // weight update function

// history arrays
activation_t   * w_output_history;  // history array for outputs
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
  uint pkt_fwbk = 0;  // unused packets received in FORWARD phase
  uint pkt_bwbk = 0;  // unused packets received in BACKPROP phase
  uint spk_sent = 0;  // sync packets sent
  uint spk_recv = 0;  // sync packets received
  uint stp_sent = 0;  // stop packets sent
  uint stp_recv = 0;  // stop packets received
  uint stn_recv = 0;  // network_stop packets received
  uint lda_sent = 0;  // partial link_delta packets sent
  uint ldr_recv = 0;  // link_delta packets received
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
  io_printf (IO_BUF, "weight\n");

  // read the data specification header
  address_t data_address = data_specification_get_data_address ();
  if (!data_specification_read_header (data_address)) {
	  rt_error (RTE_SWERR);
  }

  // get addresses of all SDRAM regions
  // network configuration address
  address_t nt = data_specification_get_region (NETWORK, data_address);

  // initialize network configuration from SDRAM
  spin1_memcpy (&ncfg, nt, sizeof (network_conf_t));

  // core configuration address
  address_t dt = data_specification_get_region (CORE, data_address);

  // initialize core-specific configuration from SDRAM
  spin1_memcpy (&wcfg, dt, sizeof (w_conf_t));

  // initial connection weights
  wt = (weight_t *) data_specification_get_region
		  (WEIGHTS, data_address);

  // examples
  ex = (struct mlp_example *) data_specification_get_region
		  (EXAMPLES, data_address);

  // routing keys
  rt = (uint *) data_specification_get_region
		  (ROUTING, data_address);

  #ifdef DEBUG_CFG0
    io_printf (IO_BUF, "nr: %d\n", wcfg.num_rows);
    io_printf (IO_BUF, "nc: %d\n", wcfg.num_cols);
    io_printf (IO_BUF, "rb: %d\n", wcfg.row_blk);
    io_printf (IO_BUF, "cb: %d\n", wcfg.col_blk);
    io_printf (IO_BUF, "lr: %k\n", wcfg.learningRate);
    io_printf (IO_BUF, "wd: %k\n", wcfg.weightDecay);
    io_printf (IO_BUF, "mm: %k\n", wcfg.momentum);
    io_printf (IO_BUF, "fk: 0x%08x\n", rt[FWD]);
    io_printf (IO_BUF, "bk: 0x%08x\n", rt[BKP]);
    io_printf (IO_BUF, "sk: 0x%08x\n", rt[FDS]);
    io_printf (IO_BUF, "ld: 0x%08x\n", rt[LDS]);
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
  uint rcode = w_init ();

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
      rt_error(RTE_SWERR);
      break;

    case SPINN_MEM_UNAVAIL:
      io_printf (IO_BUF, "malloc failed\n");
      rt_error(RTE_SWERR);
      break;

    case SPINN_UNXPD_PKT:
      io_printf (IO_BUF, "unexpected packet received - abort!\n");
      rt_error(RTE_SWERR);
      break;

    case SPINN_TIMEOUT_EXIT:
      io_printf (IO_BUF, "timeout (h:%u e:%u p:%u t:%u) - abort!\n",
                 epoch, example, phase, tick
                );

      #ifdef DEBUG_TO
        io_printf (IO_BUF, "(fp:%u  fc:%u)\n", wf_procs, wf_comms);
        io_printf (IO_BUF, "(fptd:%u)\n", wf_thrds_pend);

        io_printf (IO_BUF, "(fa:%u ba:%u)\n",
                   wf_arrived, wb_arrived
                  );
      #endif

      break;
  }

  // report diagnostics
  #ifdef DEBUG
    io_printf (IO_BUF, "total ticks:%d\n", tot_tick);
    io_printf (IO_BUF, "total recv:%d\n", pkt_recv);
    io_printf (IO_BUF, "total sent:%d\n", pkt_sent);
    io_printf (IO_BUF, "recv: fwd:%d bkp:%d\n", recv_fwd, recv_bkp);
    io_printf (IO_BUF, "sent: fwd:%d bkp:%d\n", sent_fwd, sent_bkp);
    io_printf (IO_BUF, "unused recv: fwd:%d bkp:%d\n", pkt_fwbk, pkt_bwbk);
    io_printf (IO_BUF, "sync sent:%d\n", spk_sent);
    io_printf (IO_BUF, "ldsa sent:%d\n", lda_sent);
    io_printf (IO_BUF, "ldsr recv:%d\n", ldr_recv);
    io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
    io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
    if (wrng_phs) io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
    if (wrng_tck) io_printf (IO_BUF, "wrong tick:%d\n", wrng_tck);
    if (wrng_btk) io_printf (IO_BUF, "wrong btick:%d\n", wrng_btk);
    io_printf (IO_BUF, "------\n");
    io_printf (IO_BUF, "weight updates:%d\n", wght_ups);
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
  spin1_callback_on (MC_PACKET_RECEIVED, w_receivePacket, SPINN_PACKET_P);
  spin1_callback_on (MCPL_PACKET_RECEIVED, w_receivePacket, SPINN_PACKET_P);

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
