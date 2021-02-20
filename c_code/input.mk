# input core makefile

# The name of the application to be built
APP = input

# Directory to create APLX files in (must include trailing slash)
APP_OUTPUT_DIR = ../binaries/

SOURCE_DIRS = .
SOURCES = input.c comms_i.c process_i.c init_i.c activation.c

LIBRARIES += -lm
CFLAGS += -Wno-shift-negative-value

# The GFE application standard makefile
include $(SPINN_DIRS)/make/local.mk

all: $(APP_OUTPUT_DIR)$(APP).aplx

# Tidy up
tidy:
	$(RM) $(OBJECTS) $(BUILD_DIR)$(APP).elf $(BUILD_DIR)$(APP).txt
