// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_s.h"

// this files contains the initialization routine for S cores

// ------------------------------------------------------------------------
// allocate memory and initialize variables
// ------------------------------------------------------------------------
uint s_init (void)
{
  uint i;

  // allocate memory for nets
  if ((s_nets[0] = ((long_net_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_nets[1] = ((long_net_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for errors
  if ((s_errors[0] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((s_errors[1] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for the first tick of errors in BACKPROP phase
  //TODO: is this necessary? -- not used anywhere
  if ((s_init_err[0] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  //TODO: is this necessary? -- not used anywhere
  if ((s_init_err[1] = ((long_error_t *)
         spin1_malloc (scfg.num_nets * sizeof(long_error_t)))) == NULL
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
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sf_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received error b-d-ps scoreboards
  if ((sb_arrived[0] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  if ((sb_arrived[1] = ((scoreboard_t *)
          spin1_malloc (scfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // intialize tick
  //NOTE: SUM cores do not have a tick 0
  tick = SPINN_S_INIT_TICK;

  // initialize nets, errors and scoreboards
  for (i = 0; i < scfg.num_nets; i++)
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

  #if SPLIT_ARR == TRUE
    sb_all_arrived = scfg.b_init_arrived + scfg.b_next_arrived; //#
  #else
    sb_all_arrived = scfg.b_all_arrived;
  #endif

  // initialize synchronization semaphores
  sf_thrds_done = 1;

  // initialize processing thread flag
  s_active = FALSE;

  // initialize packet queue
  s_pkt_queue.head = 0;
  s_pkt_queue.tail = 0;

  // initialize packet keys
  //NOTE: colour is initialized to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY (SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY (SPINN_BACKPROP);

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
