#ifndef __COMMS_S_H__
#define __COMMS_S_H__

void s_receivePacket (uint key,     uint payload);
void s_processQueue  (uint unused0, uint unused1);

void s_stop_packet     (uint key);
void s_net_stop_packet (uint key);

void s_ldsa_packet     (uint payload);
void s_ldst_packet     (uint payload);

#endif
