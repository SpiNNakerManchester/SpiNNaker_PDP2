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

#ifndef __COMMS_T_H__
#define __COMMS_T_H__

void t_receiveDataPacket    (uint key,     uint payload);
void t_receiveControlPacket (uint key,     uint unused);
void t_handleBKPPacket      (uint key,     uint payload);
void t_processFWDQueue      (uint unused0, uint unused1);

void t_criterion_packet (uint key);
void t_fsgn_packet      (void);
void t_stop_packet      (uint key);
void t_net_stop_packet  (uint key);
void t_dlrv_packet      (void);

void t_backprop_packet (uint key, uint payload);
void t_bsgn_packet     (void);
void t_sync_packet     (void);

void send_stop_crit (void);

void store_net            (uint inx);
void restore_net          (uint inx, uint tick);
void store_output         (uint inx);
void restore_output       (uint inx, uint tick);
void restore_outputs      (uint tick);
void store_output_deriv   (uint inx);
void restore_output_deriv (uint inx, uint tick);

void record_outputs   (void);
void record_tick_data (void);

#endif

