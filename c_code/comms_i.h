#ifndef __COMMS_I_H__
#define __COMMS_I_H__

void i_receivePacket (uint key,     uint payload);
void i_processQueue  (uint unused0, uint unused1);

void i_stop_packet     (uint key);
void i_net_stop_packet (uint key);

void store_net   (uint inx);
void restore_net (uint inx, uint tick);

#endif
 
