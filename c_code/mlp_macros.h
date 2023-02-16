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

#ifndef __MLP_MACROS_H__
#define __MLP_MACROS_H__


// ------------------------------------------------------------------------
// convenient macros for printing
// ------------------------------------------------------------------------
#define SPINN_CONV_TO_PRINT(num, shift)  num << (SPINN_PRINT_SHIFT - shift)
#define SPINN_LCONV_TO_PRINT(num, shift) num >> (shift - SPINN_PRINT_SHIFT)
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// find the absolute value of a number
// ------------------------------------------------------------------------
#define ABS(x) (((x) >= 0) ? (x) : -(x))
// ------------------------------------------------------------------------

#endif
