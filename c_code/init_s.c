// SpiNNaker API
#include "spin1_api.h"

// graph-front-end
#include <simulation.h>

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_s.h"

// this file contains the initialisation routine for S cores

// ------------------------------------------------------------------------
// allocate memory and initialise variables
// ------------------------------------------------------------------------
uint s_init (void)
{
  uint i;

  // allocate memory for nets
  if ((s_nets[0] = ((long_net_t *)
         spin1_malloc (scfg.num_units * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_nets[1] = ((long_net_t *)
         spin1_malloc (scfg.num_units * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((s_errors[0] = ((long_error_t *)
         spin1_malloc (scfg.num_units * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_errors[1] = ((long_error_t *)
         spin1_malloc (scfg.num_units * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((s_pkt_queue.queue = ((packet_t *)
         spin1_malloc (SPINN_SUM_PQ_LEN * sizeof(packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received net b-d-ps scoreboards
  if ((sf_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sf_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received error b-d-ps scoreboards
  if ((sb_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sb_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_units * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // intialize tick
  //NOTE: SUM cores do not have a tick 0
  tick = SPINN_S_INIT_TICK;

  // initialise nets, errors and scoreboards
  for (i = 0; i < scfg.num_units; i++)
  {
    s_nets[0][i] = 0;
    s_nets[1][i] = 0;
    s_errors[0][i] = 0;
    s_errors[1][i] = 0;
    sf_arrived[0][i] = 0;
    sf_arrived[1][i] = 0;
    sb_arrived[0][i] = 0;
    sb_arrived[1][i] = 0;
  }
  sf_done = 0;
  sb_done = 0;
  s_ldsa_arrived = 0;
  s_ldst_arrived = 0;

  // initialise synchronisation semaphores
  sf_thrds_pend = 1;
  sb_thrds_pend = 0;

  // initialise processing thread flag
  s_active = FALSE;

  // initialise partial lds
  s_lds_part = 0;

  // initialise packet queue
  s_pkt_queue.head = 0;
  s_pkt_queue.tail = 0;

  // initialise packet keys
  //NOTE: colour is initialised to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY (SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY (SPINN_BACKPROP);
  ldstKey = rt[LDS] | SPINN_LDST_KEY;
  ldsrKey = rt[LDS] | SPINN_LDSR_KEY;

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// stage start callback: get stage started
// ------------------------------------------------------------------------
void stage_start (void)
{
  // start log
  io_printf (IO_BUF, "--------------\n");
  io_printf (IO_BUF, "starting stage\n");
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// check exit code and print details of the state
// ------------------------------------------------------------------------
void stage_done (uint ec)
{
  // pause timer and setup next stage,
  simulation_handle_pause_resume (NULL);

  // report problems -- if any
  switch (ec)
  {
    case SPINN_NO_ERROR:
      io_printf (IO_BUF, "stage OK\n");
      break;

    case SPINN_CFG_UNAVAIL:
      io_printf (IO_BUF, "core configuration failed\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_QUEUE_FULL:
      io_printf (IO_BUF, "packet queue full\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_MEM_UNAVAIL:
      io_printf (IO_BUF, "malloc failed\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_UNXPD_PKT:
      io_printf (IO_BUF, "unexpected packet received - abort!\n");
      io_printf (IO_BUF, "stage aborted\n");
      break;

    case SPINN_TIMEOUT_EXIT:
      io_printf (IO_BUF, "timeout (h:%u e:%u p:%u t:%u) - abort!\n",
                  epoch, example, phase, tick
                );
      io_printf (IO_BUF, "stage aborted\n");
#ifdef DEBUG_TO
      io_printf (IO_BUF, "(fd:%u bd:%u)\n", sf_done, sb_done);
      for (uint i = 0; i < scfg.num_units; i++)
      {
        io_printf (IO_BUF, "%2d: (fa[0]:%u ba[0]:%u fa[1]:%u ba[1]:%u)\n", i,
                    sf_arrived[0][i], sb_arrived[0][i],
                    sf_arrived[1][i], sb_arrived[1][i]
                  );
      }
#endif
      break;
  }

#ifdef DEBUG
  // report diagnostics
  io_printf (IO_BUF, "total ticks:%d\n", tot_tick);
  io_printf (IO_BUF, "total recv:%d\n", pkt_recv);
  io_printf (IO_BUF, "total sent:%d\n", pkt_sent);
  io_printf (IO_BUF, "recv: fwd:%d bkp:%d\n", recv_fwd, recv_bkp);
  io_printf (IO_BUF, "sent: fwd:%d bkp:%d\n", sent_fwd, sent_bkp);
  io_printf (IO_BUF, "ldsa recv:%d\n", lda_recv);
  if (scfg.is_first_group)
  {
    io_printf (IO_BUF, "ldst recv:%d\n", ldt_recv);
    io_printf (IO_BUF, "ldsr sent:%d\n", ldr_sent);
  }
  else
  {
    io_printf (IO_BUF, "ldst sent:%d\n", ldt_sent);
  }
  io_printf (IO_BUF, "stop recv:%d\n", stp_recv);
  io_printf (IO_BUF, "stpn recv:%d\n", stn_recv);
  if (wrng_phs) io_printf (IO_BUF, "wrong phase:%d\n", wrng_phs);
  if (wrng_tck) io_printf (IO_BUF, "wrong tick:%d\n", wrng_tck);
  if (wrng_btk) io_printf (IO_BUF, "wrong btick:%d\n", wrng_btk);
#endif

  // close log,
  io_printf (IO_BUF, "stopping stage\n");
  io_printf (IO_BUF, "--------------\n");

  // and let host know that we're done
  if (ec == SPINN_NO_ERROR) {
    simulation_ready_to_read ();
  } else {
    rt_error (RTE_SWERR);
  }
}
// ------------------------------------------------------------------------
