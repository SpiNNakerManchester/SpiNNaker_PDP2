#ifndef __INIT_I_H__
#define __INIT_I_H__

uint cfg_init (void);
uint mem_init (void);
void var_init (void);

uint init_in_integr (void);

void stage_var_init (void);
void stage_init     (void);
void stage_start    (void);
void stage_done     (uint exit_code);

#endif
