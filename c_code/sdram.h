#ifndef __SDRAM_H__
#define __SDRAM_H__


// ------------------------------------------------------------------------
// SDRAM variables -- some need to be initialized by the host!
// ------------------------------------------------------------------------
//TODO: should be a run-time option, not a compile-time one!
#if SPINN_WEIGHT_HISTORY == TRUE
  // weight history
  static weight_t         * const wh = (weight_t *)
                                     (SPINN_SDRAM_BASE + SPINN_WUPD_OFFSET);
#endif


//TODO: should be a run-time option, not a compile-time one!
#if SPINN_OUTPUT_HISTORY == TRUE
  // unit outputs history
  static short_activ_t    * const oh = (short_activ_t *)
                                     (SPINN_SDRAM_BASE + SPINN_OUTP_OFFSET);
#endif


#endif
