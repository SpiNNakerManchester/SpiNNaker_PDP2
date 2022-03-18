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

#ifndef __COMMS_I_H__
#define __COMMS_I_H__

void i_receiveDataPacket    (uint key,     uint payload);
void i_receiveControlPacket (uint key,     uint unused);
void i_processQueue         (uint unused0, uint unused1);

void i_stop_packet     (uint key);
void i_sync_packet     (void);
void i_net_stop_packet (uint key);
void i_dlrv_packet     (void);

void store_net    (uint inx);
void restore_net  (uint inx, uint tick);
void restore_nets (uint tick);

#endif
 
