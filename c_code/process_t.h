#ifndef __PROCESS_T_H__
#define __PROCESS_T_H__

void tf_process (uint key,     uint payload);
void tb_process (uint unused0, uint unused1);

void tf_advance_tick   (void);
void tb_advance_tick   (uint unused0, uint unused1);
void tf_advance_event  (void);
void t_advance_example (void);
void t_switch_to_fw    (void);
void t_switch_to_bp    (void);

void compute_out         (uint inx);
void out_logistic        (uint inx);
void out_integr          (uint inx);
void out_hard_clamp      (uint inx);
void out_weak_clamp      (uint inx);
void out_bias            (uint inx);

void compute_out_back    (uint inx);
void out_logistic_back   (uint inx);
void out_integr_back     (uint inx);
void out_hard_clamp_back (uint inx);
void out_weak_clamp_back (uint inx);
void out_bias_back       (uint inx);

void std_stop_crit       (uint inx);
void max_stop_crit       (uint inx);

void error_cross_entropy (uint inx);
void error_squared       (uint inx);

#endif
