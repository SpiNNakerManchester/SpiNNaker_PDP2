#ifndef __COMMS_T_H__
#define __COMMS_T_H__

void t_receivePacket     (uint, uint);
void t_forwardPacket     (uint, uint);
void t_backpropPacket    (uint, uint);
void t_stopPacket        (uint);
void t_chainPacket       (uint);
void t_networkStopPacket (uint);

void record_outputs   (uint, uint);

#endif

