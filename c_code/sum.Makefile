# An example Makefile suitable for use in SpiNNaker applications using
# the spinnaker_tools libraries and makefiles.

# The name of the application to be built
# (binary will be this with a `.aplx` extension)
APP = sum

# Directory to create APLX files in (must include trailing slash)
APP_OUTPUT_DIR = ../binaries/

# Directory to place compilation artifacts (must include trailing slash)
BUILD_DIR = ./build/

CFLAGS += -Ofast
LFLAGS += -Ofast

LIBRARIES += -lm

SOURCE_DIRS = .
SOURCES = sum.c comms_s.c process_s.c init_s.c

# The spinnaker_tools standard makefile
include $(SPINN_DIRS)/make/Makefile.SpiNNFrontEndCommon

all: $(APP_OUTPUT_DIR)$(APP).aplx

# Tidy up
tidy:
	$(RM) $(OBJECTS) $(BUILD_DIR)$(APP).elf $(BUILD_DIR)$(APP).txt
