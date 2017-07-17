// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "comms_i.h"

// this files contains the initialization routine for I cores

// ------------------------------------------------------------------------
// allocate memory and initialize variables
// ------------------------------------------------------------------------
uint i_init (void)
{
  uint i;

  // allocate memory for nets
  if ((i_nets = ((long_net_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_net_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for deltas
  if ((i_deltas = ((long_delta_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // TODO: probably this variable can be removed
  // allocate memory to store delta values during the first BACKPROPagation tick
  if ((i_init_delta = ((long_delta_t *)
         spin1_malloc (icfg.num_nets * sizeof(long_delta_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for packet queue
  if ((i_pkt_queue.queue = ((packet_t *)
         spin1_malloc (SPINN_INPUT_PQ_LEN * sizeof(packet_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received net b-d-ps scoreboards
  if ((if_arrived = ((scoreboard_t *)
          spin1_malloc (icfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // allocate memory for received delta b-d-ps scoreboards
  if ((ib_arrived = ((scoreboard_t *)
          spin1_malloc (icfg.num_nets * sizeof(scoreboard_t)))) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // intialize tick
  //NOTE: input cores do not have a tick 0
  tick = SPINN_I_INIT_TICK;

  // initialize nets, deltas and scoreboards
  for (i = 0; i < icfg.num_nets; i++)
  {
    i_nets[i] = 0;
    i_deltas[i] = 0;
    if_arrived[i] = 0;
    ib_arrived[i] = 0;
  }
  if_done = 0;
  ib_done = 0;

  #if SPLIT_ARR == TRUE
    ib_all_arrived = icfg.b_init_arrived + icfg.b_next_arrived; //#
  #else
    ib_all_arrived = icfg.b_all_arrived;
  #endif

  // initialize synchronization semaphores
  if_thrds_done = 1;

  // initialize processing thread flag
  i_active = FALSE;

  // initialize packet queue
  i_pkt_queue.head = 0;
  i_pkt_queue.tail = 0;

  // initialize packet keys
  //NOTE: colour is initialized to 0.
  fwdKey = rt[FWD] | SPINN_PHASE_KEY(SPINN_FORWARD);
  bkpKey = rt[BKP] | SPINN_PHASE_KEY(SPINN_BACKPROP);

  // if input or output group initialize event input/target index
  if (icfg.input_grp || icfg.output_grp)
  {
    i_it_idx = ev[event_idx].it_idx * icfg.num_nets;
  }

  // if the network requires training and elements of the pipeline require
  // initialization, then follow the appropriate procedure
  // use the list of procedures in use from lens and call the appropriate
  // initialization routine from the i_init_in_procs function pointer list

  for (i = 0; i < icfg.num_in_procs; i++)
    if (i_init_in_procs[icfg.procs_list[i]] != NULL)
    {
      int return_value;
      // call the appropriate routine for pipeline initialization
      return_value = i_init_in_procs[icfg.procs_list[i]]();

      // if return value contains error, return it
      if (return_value != SPINN_NO_ERROR)
          return return_value;
    }

  // allocate memory in SDRAM for input history,
  // TODO: this needs a condition on the requirement to have input history
  // which needs to come from splens
  if ((i_net_history = ((long_net_t *)
          sark_xalloc (sv->sdram_heap,
                       icfg.num_nets * ncfg.global_max_ticks * sizeof(long_net_t),
                       0, ALLOC_LOCK)
                       )) == NULL
     )
  {
    return (SPINN_MEM_UNAVAIL);
  }

  // and initialise net history for tick 0.
  //TODO: understand why the values for tick 0 are used!
  for (uint i = 0; i < icfg.num_nets; i++)
  {
    i_net_history[i] = 0;
  }

  return (SPINN_NO_ERROR);
}
// ------------------------------------------------------------------------
