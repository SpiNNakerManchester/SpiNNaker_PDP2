#ifndef __COMMS_W_H__
#define __COMMS_W_H__

void w_receivePacket   (uint key, uint payload);
void w_handleFWDPacket (uint key, uint payload);
void w_processBKPQueue (uint unused0, uint unused1);

void w_forward_packet  (uint key, uint payload);
void w_stop_packet     (uint key);
void w_net_stop_packet (uint key);
void w_sync_packet     (void);

void w_ldsr_packet (uint payload);

void store_output    (uint index);
void restore_outputs (uint tick);

#endif
