#ifndef __COMMS_T_H__
#define __COMMS_T_H__

void t_receivePacket   (uint key,     uint payload);
void w_handleBKPPacket (uint key,     uint payload);
void t_processFWDQueue (uint unused0, uint unused1);

void t_criterion_packet (uint key);
void t_stop_packet      (uint key);
void t_net_stop_packet  (uint key);

void t_backprop_packet (uint key, uint payload);

void tf_send_stop (void);

void store_net            (uint inx);
void restore_net          (uint inx, uint tick);
void store_output         (uint inx);
void restore_output       (uint inx, uint tick);
void store_output_deriv   (uint inx);
void restore_output_deriv (uint inx, uint tick);

void record_outputs (void);

#endif

