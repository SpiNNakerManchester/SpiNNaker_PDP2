#ifndef __PROCESS_W_H__
#define __PROCESS_W_H__

void  wf_process        (uint null0, uint null1);
void  wb_process        (uint null0, uint null1);
void  wf_advance_tick   (uint null0, uint null1);
void  wb_advance_tick   (uint null0, uint null1);
void  wf_advance_event  (void);
void  w_advance_example (void);
void  w_switch_to_fw    (void);
void  w_switch_to_bp    (void);
void  steepest_update_weights      (void);
void  momentum_update_weights      (void);
void  dougsmomentum_update_weights (void);
void  w_weight_deltas   (void);
void  restore_outputs   (uint inx, uint tick);

#endif
