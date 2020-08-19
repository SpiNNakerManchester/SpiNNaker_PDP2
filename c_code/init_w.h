#ifndef __INIT_W_H__
#define __INIT_W_H__

uint cfg_init (void);
uint mem_init (void);
void var_init (uint init_weights, uint reset_examples);

void stage_init     (void);
void stage_start    (void);
void stage_done     (uint exit_code, uint unused);

#endif
 
