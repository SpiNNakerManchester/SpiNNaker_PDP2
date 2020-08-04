#ifndef __PROCESS_S_H__
#define __PROCESS_S_H__

void s_forward_packet  (uint key, uint payload);
void s_backprop_packet (uint key, uint payload);
void s_ldsa_packet     (uint payload);
void s_ldst_packet     (uint payload);
void s_stop_packet     (uint key);
void s_net_stop_packet (uint key);

void sf_advance_tick   (void);
void sb_advance_tick   (void);
void sf_advance_event  (void);
void s_advance_example (void);

#endif
