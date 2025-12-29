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

all: input.aplx sum.aplx threshold.aplx weight.aplx

%.aplx: %.mk src/%.c
	"$(MAKE)" -f $<

tidy:
	for d in input sum threshold weight; \
		do ("$(MAKE)" -f $$d.mk tidy) || exit $$?; done

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(abspath $(dir $(MAKEFILE_PATH)))
APP_OUTPUT_DIR := $(CURRENT_DIR)/../binaries/

clean:
	$(RM) $(APP_OUTPUT_DIR)logs.sqlite3
	$(RM) $(APP_OUTPUT_DIR)*.aplx
	$(RM) -r $(CURRENT_DIR)/build/modified_src
	for d in input sum threshold weight; \
		do ("$(MAKE)" -f $$d.mk clean) || exit $$?; done
