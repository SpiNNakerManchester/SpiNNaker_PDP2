/*
 * Copyright (c) 2015-2021 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __COMMS_W_H__
#define __COMMS_W_H__

void w_receivePacket   (uint key, uint payload);
void w_handleFWDPacket (uint key, uint payload);
void w_processBKPQueue (uint unused0, uint unused1);

void w_forward_packet  (uint key, uint payload);
void w_stop_packet     (uint key);
void w_net_stop_packet (uint key);
void w_sync_packet     (void);
void w_dlrv_packet     (void);

void w_lds_packet (uint payload);

void store_output    (uint index);
void restore_outputs (uint tick);

#endif
