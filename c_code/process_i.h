#ifndef __PROCESS_I_H__
#define __PROCESS_I_H__

void i_forward_packet  (uint key, uint payload);
void i_backprop_packet (uint key, uint payload);
void i_stop_packet     (uint key);
void i_net_stop_packet (uint key);

void if_advance_tick   (void);
void ib_advance_tick   (void);
void if_advance_event  (void);
void i_advance_example (void);

void store_nets        (uint inx);
void restore_nets      (uint inx, uint tick);

void compute_in        (uint inx);
void in_integr         (uint inx);
void in_soft_clamp     (uint inx);

void compute_in_back   (uint inx);
void in_integr_back    (uint inx);

#endif
