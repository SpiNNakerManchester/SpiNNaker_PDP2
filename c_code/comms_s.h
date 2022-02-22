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

#ifndef __COMMS_S_H__
#define __COMMS_S_H__

void s_receivePacket (uint key,     uint payload);
void s_processQueue  (uint unused0, uint unused1);

void s_fsgn_packet     (void);
void s_stop_packet     (uint key);
void s_net_stop_packet (uint key);
void s_dlrv_packet     (void);

void s_lds_packet  (uint payload);
void s_bsgn_packet (void);
void s_sync_packet (void);

#endif
