# threshold core makefile

# The name of the application to be built
APP = threshold

# Directory to create APLX files in (must include trailing slash)
APP_OUTPUT_DIR = ../binaries/

SOURCE_DIRS = .
SOURCES = threshold.c comms_t.c process_t.c init_t.c activation.c

LIBRARIES += -lm

# The GFE application standard makefile
include $(SPINN_DIRS)/make/local.mk

all: $(APP_OUTPUT_DIR)$(APP).aplx

# Tidy up
tidy:
	$(RM) $(OBJECTS) $(BUILD_DIR)$(APP).elf $(BUILD_DIR)$(APP).txt
