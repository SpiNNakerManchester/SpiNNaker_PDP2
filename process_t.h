#ifndef __PROCESS_T_H__
#define __PROCESS_T_H__

void  tf_process        (uint null0, uint null1);
void  tb_process        (uint null0, uint null1);
void  tf_advance_tick   (uint null0, uint null1);
void  tb_advance_tick   (uint null0, uint null1);
void  tf_advance_event  (void);
void  t_advance_example (void);
void  t_switch_to_fw    (void);
void  t_switch_to_bp    (void);
void  tf_send_stop      (uint null0, uint null1);
void  t_init_outputs    (uint null0, uint null1);

void  store_outputs	  (void);
void  store_targets	  (void);
void  store_output_deriv  (void);
void  restore_output_deriv (uint inx);

void  compute_out         (uint inx);
void  out_logistic        (uint inx);
void  out_integr          (uint inx);
void  out_hard_clamp      (uint inx);
void  out_weak_clamp      (uint inx);
void  out_bias            (uint inx);

void  compute_out_back    (uint inx);
void  out_logistic_back   (uint inx);
void  out_integr_back     (uint inx);
void  out_hard_clamp_back (uint inx);
void  out_weak_clamp_back (uint inx);
void  out_bias_back       (uint inx);

int   init_out_integr     (void);
int   init_out_hard_clamp (void);
int   init_out_weak_clamp (void);

void  std_stop_crit       (uint inx);
void  max_stop_crit       (uint inx);

void  error_cross_entropy (uint inx);
void  error_squared       (uint inx);

#endif
