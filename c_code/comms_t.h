#ifndef __COMMS_T_H__
#define __COMMS_T_H__

void t_receivePacket  (uint, uint);
void t_forwardPacket  (uint, uint);
void t_backpropPacket (uint, uint);
void t_stopPacket     (uint, uint);
void t_chainPacket    (uint, uint);
void t_syncPacket     (uint, uint);

//#void t_sendDeltas    (uint, uint);
void send_info_to_host    (uint, uint);
//#void send_outputs_to_host (ushort);
void send_outputs_to_host (uint, uint);
//#void send_weights_to_host (void);

#endif
 