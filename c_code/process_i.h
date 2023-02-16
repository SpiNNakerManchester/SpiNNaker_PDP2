/*
 * Copyright (c) 2015 The University of Manchester
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

#ifndef __PROCESS_I_H__
#define __PROCESS_I_H__

void if_process (uint key, uint payload);
void ib_process (uint key, uint payload);

void if_advance_tick   (uint unused0, uint unused1);
void ib_advance_tick   (uint unused0, uint unused1);
void if_advance_event  (void);
void i_advance_example (void);

void compute_in    (uint inx);
void in_integr     (uint inx);
void in_soft_clamp (uint inx);

void compute_in_back (uint inx);
void in_integr_back  (uint inx);

void store_net    (uint inx);
void restore_net  (uint inx, uint tick);
void restore_nets (uint tick);

#endif
