#ifndef __PROCESS_S_H__
#define __PROCESS_S_H__

void sf_process (uint key, uint payload);
void sb_process (uint key, uint payload);

void sf_advance_tick   (void);
void sb_advance_tick   (void);
void sf_advance_event  (void);
void s_advance_example (void);

#endif
