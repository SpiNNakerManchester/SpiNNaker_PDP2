##------------------------------------------------------------------------------
##
## Makefile        Makefile for a simple SpiNNaker application
##
## Copyright (C)   The University of Manchester - 2013
##
## Author          Steve Temple, APT Group, School of Computer Science
##
## Email           temples@cs.man.ac.uk
##
##------------------------------------------------------------------------------

# Makefile for a simple SpiNNaker application. This will compile
# a single C source file into an APLX file which can be loaded onto
# SpiNNaker. It will link with either a 'bare' SARK library or a
# combined SARK/API library.

# The options below can be overridden from the command line or via
# environment variables. For example, to compile and link "my_example.c"
# with the ARM tools and generate ARM (as opposed to Thumb) code
#
# make APP=my_example GNU=0 THUMB=0

# Name of app (derived from C source - eg sark.c)

APP := mlp

# Configuration options

# Set to 1 for GNU tools, 0 for ARM

GNU := 0

# Set to 1 if using SARK/API (0 for SARK)

API := 1

# Set to 1 to make Thumb code (0 for ARM)

THUMB := 0

# Prefix for GNU tool binaries

GP  := arm-none-eabi

# Set to 1 if making a library (advanced!)

LIB := 0

# If SPINN_DIRS is defined, use that to find include and lib directories
# otherwise look in parent directory

ifdef SPINN_DIRS
  LIB_DIR := $(SPINN_DIRS)/lib
  INC_DIR := $(SPINN_DIRS)/include
else
  LIB_DIR := ../lib
  INC_DIR := ../include
endif

#-------------------------------------------------------------------------------

# Set up the various compile/link options for GNU and ARM tools

# GNU tool setup

ifeq ($(GNU),1)
  AS := $(GP)-as --defsym GNU=1 -mthumb-interwork -march=armv5te

  CA := $(GP)-gcc -c -Os -mthumb-interwork -march=armv5te -std=gnu99 \
	-I $(INC_DIR)

  CT := $(CA) -mthumb -DTHUMB

ifeq ($(LIB),1)
  CFLAGS += -fdata-sections -ffunction-sections
endif

ifeq ($(API),1)
#  LIBRARY := -L$(LIB_DIR) -lspin1_api
  LIBRARY := $(LIB_DIR)/libspin1_api.a
else
#  LIBRARY := -L$(LIB_DIR) -lsark
  LIBRARY := $(LIB_DIR)/libsark.a
endif

  SCRIPT := $(LIB_DIR)/sark.lnk

  LD := $(GP)-gcc -T$(SCRIPT) -Wl,-e,cpu_reset -Wl,--gc-sections -Wl,--use-blx 

  AR := $(GP)-ar -rcs
  OC := $(GP)-objcopy
  OD := $(GP)-objdump -dxt >

# ARM tool setup

else
  AS := armasm --keep --cpu=5te --apcs /interwork

  CA := armcc -c --c99 --cpu=5te --apcs /interwork --min_array_alignment=4 \
	-I $(INC_DIR)

  CT := $(CA) --thumb -DTHUMB

ifeq ($(LIB),1)
  CFLAGS += --split_sections
endif

ifeq ($(API),1)
  LIBRARY := $(LIB_DIR)/spin1_api.a
else
  LIBRARY := $(LIB_DIR)/sark.a
endif

  SCRIPT := $(LIB_DIR)/sark.sct

  LD := armlink --scatter=$(SCRIPT) --remove --entry cpu_reset

  AR := armar -rcs
  OC := fromelf
  OD := fromelf -cds --output

endif

ifeq ($(THUMB),1)
  CC := $(CT)
else
  CC := $(CA)
endif

CAT := \cat
RM  := \rm -f
LS  := \ls -l

#-------------------------------------------------------------------------------

# Build the application

# List of objects making up the application. If there are other files
# in the application, add their object file names to this variable.

# OBJECTS := input.o sum.o threshold.o weight.o init_w.o init_s.o init_i.o init_t.o comms_w.o comms_s.o comms_i.o comms_t.o activation.o process_w.o process_s.o process_i.o process_t.o
OBJECTS_I := input.o init_i.o comms_i.o activation.o process_i.o
OBJECTS_S := sum.o init_s.o comms_s.o process_s.o
OBJECTS_T := threshold.o init_t.o comms_t.o activation.o process_t.o
OBJECTS_W := weight.o init_w.o comms_w.o process_w.o

# Primary target is an APLX file - built from the ELF

#  1) Create a binary file which is the concatenation of RO and RW sections
#  2) Make an APLX header from the ELF file with "mkaplx" and concatenate
#     that with the binary to make the APLX file
#  3) Remove temporary files and "ls" the APLX file

$(APP): input.aplx sum.aplx threshold.aplx weight.aplx
	$(LS) *.aplx

input: input.aplx

sum: sum.aplx

threshold: threshold.aplx

weight: weight.aplx


input.aplx: input.elf
ifeq ($(GNU),1)
	$(OC) -O binary -j RO_DATA input.elf RO_DATA.bin
	$(OC) -O binary -j RW_DATA input.elf RW_DATA.bin
	mkbin RO_DATA.bin RW_DATA.bin > input.bin
else
	$(OC) --bin --output input.bin input.elf
endif
	mkaplx input.elf | $(CAT) - input.bin > input.aplx
	$(RM) input.bin RO_DATA.bin RW_DATA.bin
	$(LS) input.aplx

sum.aplx: sum.elf
ifeq ($(GNU),1)
	$(OC) -O binary -j RO_DATA sum.elf RO_DATA.bin
	$(OC) -O binary -j RW_DATA sum.elf RW_DATA.bin
	mkbin RO_DATA.bin RW_DATA.bin > sum.bin
else
	$(OC) --bin --output sum.bin sum.elf
endif
	mkaplx sum.elf | $(CAT) - sum.bin > sum.aplx
	$(RM) sum.bin RO_DATA.bin RW_DATA.bin
	$(LS) sum.aplx

threshold.aplx: threshold.elf
ifeq ($(GNU),1)
	$(OC) -O binary -j RO_DATA threshold.elf RO_DATA.bin
	$(OC) -O binary -j RW_DATA threshold.elf RW_DATA.bin
	mkbin RO_DATA.bin RW_DATA.bin > threshold.bin
else
	$(OC) --bin --output threshold.bin threshold.elf
endif
	mkaplx threshold.elf | $(CAT) - threshold.bin > threshold.aplx
	$(RM) threshold.bin RO_DATA.bin RW_DATA.bin
	$(LS) threshold.aplx

weight.aplx: weight.elf
ifeq ($(GNU),1)
	$(OC) -O binary -j RO_DATA weight.elf RO_DATA.bin
	$(OC) -O binary -j RW_DATA weight.elf RW_DATA.bin
	mkbin RO_DATA.bin RW_DATA.bin > weight.bin
else
	$(OC) --bin --output weight.bin weight.elf
endif
	mkaplx weight.elf | $(CAT) - weight.bin > weight.aplx
	$(RM) weight.bin RO_DATA.bin RW_DATA.bin
	$(LS) weight.aplx


# Build the ELF file

#  1) Make a "sark_build.c" file containing app. name and build time
#     with "mkbuild" and compile it
#  2) Link application object(s), build file and library to make the ELF
#  3) Tidy up temporaries and create a list file

input.elf: $(OBJECTS_I) $(SCRIPT) $(LIBRARY) 
	mkbuild input > sark_build.c
	$(CC) sark_build.c
	$(LD) $(LFLAGS) $(OBJECTS_I) sark_build.o $(LIBRARY) -o input.elf
	$(RM) sark_build.c sark_build.o
	$(OD) input.txt input.elf

sum.elf: $(OBJECTS_S) $(SCRIPT) $(LIBRARY) 
	mkbuild sum > sark_build.c
	$(CC) sark_build.c
	$(LD) $(LFLAGS) $(OBJECTS_S) sark_build.o $(LIBRARY) -o sum.elf
	$(RM) sark_build.c sark_build.o
	$(OD) sum.txt sum.elf

threshold.elf: $(OBJECTS_T) $(SCRIPT) $(LIBRARY) 
	mkbuild threshold > sark_build.c
	$(CC) sark_build.c
	$(LD) $(LFLAGS) $(OBJECTS_T) sark_build.o $(LIBRARY) -o threshold.elf
	$(RM) sark_build.c sark_build.o
	$(OD) threshold.txt threshold.elf

weight.elf: $(OBJECTS_W) $(SCRIPT) $(LIBRARY) 
	mkbuild weight > sark_build.c
	$(CC) sark_build.c
	$(LD) $(LFLAGS) $(OBJECTS_W) sark_build.o $(LIBRARY) -o weight.elf
	$(RM) sark_build.c sark_build.o
	$(OD) weight.txt weight.elf

# Build the main object file. If there are other files in the
# application, place their build dependencies below this one.

input.o: input.c $(INC_DIR)/spinnaker.h $(INC_DIR)/sark.h \
	  $(INC_DIR)/spin1_api.h mlp_types.h mlp_params.h
	$(CC) $(CFLAGS) input.c

sum.o: sum.c $(INC_DIR)/spinnaker.h $(INC_DIR)/sark.h \
	  $(INC_DIR)/spin1_api.h mlp_types.h mlp_params.h
	$(CC) $(CFLAGS) sum.c

threshold.o: threshold.c $(INC_DIR)/spinnaker.h $(INC_DIR)/sark.h \
	  $(INC_DIR)/spin1_api.h mlp_types.h mlp_params.h
	$(CC) $(CFLAGS) threshold.c

weight.o: weight.c $(INC_DIR)/spinnaker.h $(INC_DIR)/sark.h \
	  $(INC_DIR)/spin1_api.h mlp_types.h mlp_params.h
	$(CC) $(CFLAGS) weight.c

init_i.o: init_i.c init_i.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) init_i.c

init_s.o: init_s.c init_s.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) init_s.c

init_t.o: init_t.c init_t.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) init_t.c

init_w.o: init_w.c init_w.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) init_w.c

comms_i.o: comms_i.c comms_i.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) comms_i.c

comms_s.o: comms_s.c comms_s.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) comms_s.c

comms_t.o: comms_t.c comms_t.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) comms_t.c

comms_w.o: comms_w.c comms_w.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) comms_w.c

#process.o: process.c process.h mlp_types.h mlp_params.h
#	$(CC)  $(CFLAGS) process.c

process_w.o: process_w.c process_w.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) process_w.c

process_s.o: process_s.c process_s.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) process_s.c

process_i.o: process_i.c process_i.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) process_i.c

process_t.o: process_t.c process_t.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) process_t.c

activation.o: activation.c activation.h mlp_types.h mlp_params.h
	$(CC)  $(CFLAGS) activation.c

# Tidy and cleaning dependencies

tidy:
	$(RM) $(OBJECTS_I) $(OBJECTS_S) $(OBJECTS_T) $(OBJECTS_W) input.elf sum.elf threshold.elf weight.elf input.txt sum.txt threshold.txt weight.txt *~

clean: tidy
	$(RM) input.aplx sum.aplx threshold.aplx weight.aplx

#-------------------------------------------------------------------------------
