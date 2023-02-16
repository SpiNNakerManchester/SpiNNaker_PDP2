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

#ifndef __INIT_W_H__
#define __INIT_W_H__

uint cfg_init  (void);
uint mem_init  (void);
void tick_init (uint restart,      uint unused);
void var_init  (uint init_weights, uint reset_examples);

void timeout_rep (uint abort);

void stage_init     (void);
void stage_start    (void);
void stage_done     (uint exit_code, uint unused);

#endif
 
