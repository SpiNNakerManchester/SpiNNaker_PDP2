// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_externs.h"

#include "init_s.h"
#include "comms_s.h"
#include "process_s.h"


// ------------------------------------------------------------------------
// sum core communications routines
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// enqueue received packet
// (FORWARD, BACKPROP, ldsa, ldst, stop and net_stop types)
// ------------------------------------------------------------------------
void s_receivePacket (uint key, uint payload)
{
#ifdef DEBUG
  pkt_recv++;
#endif

  // queue packet - if space available
  uint new_tail = (s_pkt_queue.tail + 1) % SPINN_SUM_PQ_LEN;
  if (new_tail == s_pkt_queue.head)
  {
      // report queue full error
      stage_done (SPINN_QUEUE_FULL, 0);
  }
  else
  {
    // if not full enqueue packet,
    s_pkt_queue.queue[s_pkt_queue.tail].key = key;
    s_pkt_queue.queue[s_pkt_queue.tail].payload = payload;
    s_pkt_queue.tail = new_tail;

    // and schedule processing thread -- if not active already
    if (!s_active)
    {
      s_active = TRUE;
      spin1_schedule_callback (s_processQueue, 0, 0, SPINN_S_PROCESS_P);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process packet queue until empty
// ------------------------------------------------------------------------
void s_processQueue (uint unused0, uint unused1)
{
  (void) unused0;
  (void) unused1;

#ifdef TRACE
  io_printf (IO_BUF, "s_process\n");
#endif

  // access queue with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // process until queue empty,
  while (s_pkt_queue.head != s_pkt_queue.tail)
  {
    // if not empty dequeue packet,
    uint key = s_pkt_queue.queue[s_pkt_queue.head].key;
    uint payload = s_pkt_queue.queue[s_pkt_queue.head].payload;
    s_pkt_queue.head = (s_pkt_queue.head + 1) % SPINN_SUM_PQ_LEN;

    // restore interrupts after queue access,
    spin1_mode_restore (cpsr);

    uint pkt_type = key & SPINN_TYPE_MASK;

    // check if data packet,
    if (pkt_type == SPINN_DATA_KEY)
    {
      // check packet phase and process accordingly
      uint ph = (key & SPINN_PHASE_MASK) >> SPINN_PHASE_SHIFT;

      if (ph == SPINN_FORWARD)
      {
        // process FORWARD phase packet
        sf_process (key, payload);
      }
      else
      {
        // process BACKPROP phase packet
        sb_process (key, payload);
      }
    }

    // check for an LDS "accumulation" packet,
    else if (pkt_type == SPINN_LDSA_KEY)
    {
      // process LDS "accumulation" packet
      s_ldsa_packet (payload);
    }

    // check for LDS "total" packet,
    else if (pkt_type == SPINN_LDST_KEY)
    {
      // process LDS "total" packet
      s_ldst_packet (payload);
    }

    // check if stop packet,
    else if (pkt_type == SPINN_STOP_KEY)
    {
      // stop packet received
      s_stop_packet (key);
    }

    // check if network stop packet,
    else if (pkt_type == SPINN_STPN_KEY)
    {
      // network stop packet received
      s_net_stop_packet (key);
    }

#ifdef DEBUG
    // report unknown packet type,
    else
    {
      stage_done (SPINN_UNXPD_PKT, key);
    }
#endif

    // and access queue with interrupts disabled
    cpsr = spin1_int_disable ();
  }

  // flag going to sleep,
  s_active = FALSE;

  // and restore interrupts
  spin1_mode_restore (cpsr);
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a tick stop packet
// ------------------------------------------------------------------------
void s_stop_packet (uint key)
{
#ifdef DEBUG
  stp_recv++;
#endif

  // tick stop decision arrived,
  tick_stop = key & SPINN_STPD_MASK;

  // access thread semaphore with interrupts disabled,
  uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
  if (!(sf_thrds_pend & SPINN_THRD_STOP))
    wrng_sth++;
#endif

  // and check if all other threads done
  if (sf_thrds_pend == SPINN_THRD_STOP)
  {
    // if done initialise semaphore,
    sf_thrds_pend = SPINN_SF_THRDS;

    // restore interrupts after semaphore access,
    spin1_mode_restore (cpsr);

    // and advance tick
    sf_advance_tick ();
  }
  else
  {
    // if not done report processing thread done,
    sf_thrds_pend &= ~SPINN_THRD_STOP;

    // and restore interrupts after semaphore access
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process a network stop packet
// ------------------------------------------------------------------------
void s_net_stop_packet (uint key)
{
#ifdef DEBUG
  stn_recv++;
#endif

  // network stop decision arrived,
  net_stop = key & SPINN_STPD_MASK;

  // access flag with interrupts disabled,
  uint cpsr = spin1_int_disable ();

  // and check if ready for network stop decision
  if (net_stop_rdy)
  {
    // clear flag,
    net_stop_rdy = FALSE;

    // restore interrupts after flag access,
    spin1_mode_restore (cpsr);

    // and decide what to do
    if (net_stop)
    {
      // finish stage and report no error
      //TODO: check if need to schedule or can simply call
      spin1_schedule_callback (stage_done, SPINN_NO_ERROR, 0, SPINN_DONE_P);
    }
  }
  else
  {
    // flag ready for net_stop decision,
    net_stop_rdy = TRUE;

    // and restore interrupts after flag access,
    spin1_mode_restore (cpsr);
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process LDSA packet: accumulate the received partial link delta sums
// ------------------------------------------------------------------------
void s_ldsa_packet (uint payload)
{
#ifdef DEBUG
  lda_recv++;
#endif

  // add the received value to the total so far,
  s_lds_part += (lds_t) payload;

  // increment the count of partial link delta sums arrived,
  s_ldsa_arrived++;

  // check whether all the partial sums have arrived
  if (s_ldsa_arrived == scfg.ldsa_expected)
  {
    // send the result to the first s core
    // to give a total across the whole network
    if (scfg.is_first_group == 0)
    {
      while (!spin1_send_mc_packet (ldstKey, s_lds_part, WITH_PAYLOAD));

#ifdef DEBUG
      pkt_sent++;
      ldt_sent++;
#endif
    }

    // access thread semaphore with interrupts disabled
    uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(sb_thrds_pend & SPINN_THRD_LDSA))
      wrng_cth++;
#endif

    // check if all other threads done
    if (sb_thrds_pend == SPINN_THRD_LDSA)
    {
      // if done initialise semaphore
      sb_thrds_pend = SPINN_SB_THRDS;

      // restore interrupts after flag access,
      spin1_mode_restore (cpsr);

      // and advance tick
      sb_advance_tick ();
    }
    else
    {
      // if not done report processing thread done,
      sb_thrds_pend &= ~SPINN_THRD_LDSA;

      // and restore interrupts after flag access
      spin1_mode_restore (cpsr);
    }
  }
}
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// process LDST packet: accumulate the received link delta sum totals
// ------------------------------------------------------------------------
void s_ldst_packet (uint payload)
{
#ifdef DEBUG
  ldt_recv++;
#endif

  // add the received value to the total so far,
  s_lds_part += (lds_t) payload;

  // increment the count of link delta sums arrived,
  s_ldst_arrived++;

  // check whether all the partial sums have arrived
  if (s_ldst_arrived == scfg.ldst_expected)
  {
    // send the final value of s_lds_part back to the w cores
    while (!spin1_send_mc_packet (ldsrKey, s_lds_part, WITH_PAYLOAD));

#ifdef DEBUG
    pkt_sent++;
    ldr_sent++;
#endif

    // access thread semaphore with interrupts disabled
    uint cpsr = spin1_int_disable ();

#if defined(DEBUG) && defined(DEBUG_THRDS)
    if (!(sb_thrds_pend & SPINN_THRD_LDST))
      wrng_cth++;
#endif

    // check if all other threads done
    if (sb_thrds_pend == SPINN_THRD_LDST)
    {
      // if done initialise semaphore
      sb_thrds_pend = SPINN_SB_THRDS;

      // restore interrupts after semaphore access,
      spin1_mode_restore (cpsr);

      // and advance tick
      sb_advance_tick ();
    }
    else
    {
      // if not done report processing thread done,
      sb_thrds_pend &= ~SPINN_THRD_LDST;

      // and restore interrupts after semaphore access
      spin1_mode_restore (cpsr);
    }
  }
}
// ------------------------------------------------------------------------
