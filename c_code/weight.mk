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

# weight core makefile

FEC_INSTALL_DIR := $(strip $(if $(FEC_INSTALL_DIR), $(FEC_INSTALL_DIR), $(if $(SPINN_DIRS), $(SPINN_DIRS)/fec_install, $(error FEC_INSTALL_DIR or SPINN_DIRS is not set.  Please define FEC_INSTALL_DIR or SPINN_DIRS))))

# The name of the application to be built
APP = weight

# Directory to create APLX files in (must include trailing slash)
APP_OUTPUT_DIR = ../binaries/

SOURCES = weight.c comms_w.c process_w.c init_w.c activation.c

LIBRARIES += -lm
CFLAGS += -Wno-shift-negative-value

# The GFE application standard makefile
include $(FEC_INSTALL_DIR)/make/fec.mk
