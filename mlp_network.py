import struct
from enum import Enum

class MLPRegions (Enum):
    """ regions used by MLP cores
    """
    NETWORK = 0
    CORE = 1
    INPUTS = 2
    TARGETS = 3
    EXAMPLE_SET = 4
    EXAMPLES = 5
    EVENTS = 6
    WEIGHTS = 7
    ROUTING = 8

class MLPTypes (Enum):
    """ MLP network types
    """
    FEED_FWD   = 0
    SIMPLE_REC = 1
    RBPTT      = 2
    CONT       = 3


class MLPInputProcs (Enum):
    """ MLP input-stage procedures
    """
    IN_INTEGR     = 0
    IN_SOFT_CLAMP = 1
    IN_NONE       = 255


class MLPNetwork():
    """ top-level MLP network object.
            contains groups and links
            and top-level properties.
    """

    def __init__(self,
                net_type,
                training=None,
                num_epochs=None,
                num_examples=None,
                ticks_per_int=None,
                global_max_ticks=None,
                num_write_blks=None,
                timeout=100
                ):
        """
        """

        self._net_type = net_type
        self._training = training
        self._num_epochs = num_epochs
        self._num_examples = num_examples
        self._ticks_per_int = ticks_per_int
        self._global_max_ticks = global_max_ticks
        self._num_write_blks = num_write_blks
        self._timeout = timeout

    @property
    def net_type (self):
        return self._net_type

    @property
    def training (self):
        return self._training

    @property
    def num_epochs (self):
        return self._num_epochs

    @property
    def num_examples (self):
        return self._num_examples

    @property
    def ticks_per_int (self):
        return self._ticks_per_int

    @property
    def global_max_ticks (self):
        return self._global_max_ticks

    @property
    def num_write_blocks (self):
        return self._num_write_blks

    @property
    def timeout (self):
        return self._timeout

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) network_conf in mlp_types.h:

            typedef struct network_conf
            {
              uchar net_type;
              uchar training;
              uint  num_epochs;
              uint  num_examples;
              uint  ticks_per_int;
              uint  global_max_ticks;
              uint  num_write_blks;
              uint  timeout;
            } network_conf_t;

            pack: standard sizes, little-endian byte-order,
            explicit padding
        """
        return struct.pack("<2B2x6I",
                           self._net_type,
                           self._training,
                           self._num_epochs,
                           self._num_examples,
                           self._ticks_per_int,
                           self._global_max_ticks,
                           self._num_write_blks,
                           self._timeout
                           )
