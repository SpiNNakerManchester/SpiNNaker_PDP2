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

#ifndef __PROCESS_W_H__
#define __PROCESS_W_H__

void wf_process (uint unused0, uint unused1);
void wb_process (uint key,     uint payload);

void wf_advance_tick   (uint unused0, uint unused1);
void wb_advance_tick   (void);
void wf_advance_event  (void);
void w_advance_example (void);
void w_switch_to_fw    (void);
void w_switch_to_bp    (void);

void steepest_update_weights      (void);
void momentum_update_weights      (void);
void dougsmomentum_update_weights (void);
void w_weight_deltas              (void);

#endif
