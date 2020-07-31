#ifndef __COMMS_W_H__
#define __COMMS_W_H__

void w_receivePacket     (uint, uint);
void w_stopPacket        (uint);
void w_forwardPacket     (uint, uint);
void w_backpropPacket    (uint, uint);
void w_ldsrPacket        (uint);
void w_syncPacket        (void);
void w_networkStopPacket (uint);
void store_output        (uint);

#endif
