#ifndef __PROCESS_S_H__
#define __PROCESS_S_H__

void  s_process         (uint unused0, uint unused1);
void  s_forward_packet  (uint key,     uint payload);
void  s_backprop_packet (uint key,     uint payload);
void  s_ldsa_packet     (uint payload);
void  s_ldst_packet     (uint payload);
void  sf_advance_tick   (uint unused0, uint unused1);
void  sb_advance_tick   (uint unused0, uint unused1);
void  sf_advance_event  (void);
void  s_advance_example (void);

#endif
