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

#ifndef __PROCESS_T_H__
#define __PROCESS_T_H__

void tf_process (uint key,     uint payload);
void tb_process (uint unused0, uint unused1);

void tf_advance_tick   (uint unused0, uint unused1);
void tb_advance_tick   (uint unused0, uint unused1);
void tf_advance_event  (void);
void t_advance_example (void);
void t_switch_to_fw    (void);
void t_switch_to_bp    (void);

void compute_out         (uint inx);
void out_logistic        (uint inx);
void out_integr          (uint inx);
void out_hard_clamp      (uint inx);
void out_weak_clamp      (uint inx);
void out_bias            (uint inx);

void compute_out_back    (uint inx);
void out_logistic_back   (uint inx);
void out_integr_back     (uint inx);
void out_hard_clamp_back (uint inx);
void out_weak_clamp_back (uint inx);
void out_bias_back       (uint inx);

void std_stop_crit       (uint inx);
void max_stop_crit       (uint inx);

void error_cross_entropy (uint inx);
void error_squared       (uint inx);

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
