#ifndef __PROCESS_I_H__
#define __PROCESS_I_H__

void if_process (uint key, uint payload);
void ib_process (uint key, uint payload);

void if_advance_tick   (void);
void ib_advance_tick   (void);
void if_advance_event  (void);
void i_advance_example (void);

void compute_in    (uint inx);
void in_integr     (uint inx);
void in_soft_clamp (uint inx);

void compute_in_back (uint inx);
void in_integr_back  (uint inx);

#endif
