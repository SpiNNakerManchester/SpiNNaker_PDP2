#ifndef __PROCESS_W_H__
#define __PROCESS_W_H__

void wf_process (uint unused0, uint unused1);
void wb_process (uint key,     uint payload);

void wf_advance_tick   (uint unused0, uint unused1);
void wb_advance_tick   (void);
void wf_advance_event  (void);
void w_advance_example (void);
void w_switch_to_fw    (void);
void w_switch_to_bp    (void);

void steepest_update_weights      (void);
void momentum_update_weights      (void);
void dougsmomentum_update_weights (void);
void w_weight_deltas              (void);

#endif
