# Copyright (c) 2015 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# threshold core makefile

ifndef FEC_INSTALL_DIR:
    CUR_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
    # assume parallel clone
    FEC_INSTALL_DIR := $(abspath $(CUR_DIR)/../../SpiNNFrontEndCommon/c_common/front_end_common_lib)
endif

# The name of the application to be built
APP = threshold

# Directory to create APLX files in (must include trailing slash)
APP_OUTPUT_DIR = ../binaries/

SOURCES = threshold.c comms_t.c process_t.c init_t.c activation.c

LIBRARIES += -lm
CFLAGS += -Wno-shift-negative-value

# The GFE application standard makefile
include $(FEC_INSTALL_DIR)/make/fec.mk
