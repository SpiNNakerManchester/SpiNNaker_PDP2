from enum import Enum


class MLPUpdateFuncs (Enum):
    """ MLP weight update functions
    """
    UPD_STEEPEST      = 0
    UPD_MOMENTUM      = 1
    UPD_DOUGSMOMENTUM = 2


class MLPConstants ():
    """ MLP network constants
    """
    # network parameter CONSTANTS or DEFAULT values
    DEF_LEARNING_RATE = 0.1
    DEF_WEIGHT_DECAY = 0
    DEF_MOMENTUM = 0.9
    DEF_UPDATE_FUNC = MLPUpdateFuncs.UPD_DOUGSMOMENTUM
    DEF_NUM_UPDATES = 1

    DEF_INIT_NET  = 0
    DEF_INIT_OUT  = 0.5

    BIAS_INIT_OUT = 1.0

    MAX_IN_PROCS  = 2
    DEF_IN_PROCS  = 0

    MAX_GRP_UNITS = 128
    MAX_BLK_UNITS = 32

    MAX_OUT_PROCS = 5
    DEF_OUT_PROCS = 2

    DEF_SOFT_CLMP = 0.5
    DEF_WEAK_CLMP = 0.5

    DEF_GRP_CRIT   = 0
    DEF_EX_FREQ    = 1.0

    # core configuration CONSTANTS
    KEY_SPACE_SIZE = 65536
    NUM_KEYS_REQ   = 5

    # MLP fixed-point fpreal type CONSTANTS
    FPREAL_SIZE      = 32
    FPREAL_SHIFT     = 16
    FPREAL_NaN       = (1 << (FPREAL_SIZE - 1)) & 0xffffffff

    # MLP fixed-point short fpreal type CONSTANTS
    SHORT_FPREAL_SHIFT = 15

    # MLP fixed-point activation_t type CONSTANTS
    ACTIV_SIZE  = 32
    ACTIV_SHIFT = 27
    ACTIV_NaN   = (1 << (ACTIV_SIZE - 1)) & 0xffffffff

    # MLP fixed-point net_t type CONSTANTS
    NET_SIZE = 32
    LONG_NET_SIZE = 64

    # MLP fixed-point deriv_t type CONSTANTS
    DERIV_SIZE = 32
    LONG_DERIV_SIZE = 64

    # MLP fixed-point error_t type CONSTANTS
    ERROR_SHIFT = 15

    # MLP fixed-point weight_t type CONSTANTS
    WEIGHT_SHIFT       = 15
    WEIGHT_MAX         = 0xffff << WEIGHT_SHIFT
    WEIGHT_MIN         = -WEIGHT_MAX
    WEIGHT_POS_EPSILON = 1
    WEIGHT_NEG_EPSILON = -1

    # weights file CONSTANTS
    LENS_WEIGHT_MAGIC_COOKIE = 1431655766
    WF_MAX = (1.0 * WEIGHT_MAX) / (1.0 * (1 << WEIGHT_SHIFT))
    WF_MIN = (1.0 * WEIGHT_MIN) / (1.0 * (1 << WEIGHT_SHIFT))
    WF_EPS = (1.0 * WEIGHT_POS_EPSILON) / (1.0 * (1 << WEIGHT_SHIFT))


class MLPNetworkTypes (Enum):
    """ MLP network types
    """
    FEED_FWD   = 0
    SIMPLE_REC = 1
    RBPTT      = 2
    CONTINUOUS = 3


class MLPGroupTypes (Enum):
    """ MLP network types
    """
    BIAS   = 0
    INPUT  = 1
    OUTPUT = 2
    HIDDEN = 3


class MLPInputProcs (Enum):
    """ MLP input-stage procedures
    """
    IN_INTEGR     = 0
    IN_SOFT_CLAMP = 1
    IN_NONE       = 255


class MLPOutputProcs (Enum):
    """ MLP output-stage procedures
    """
    OUT_LOGISTIC   = 0
    OUT_INTEGR     = 1
    OUT_HARD_CLAMP = 2
    OUT_WEAK_CLAMP = 3
    OUT_BIAS       = 4
    OUT_NONE       = 255


class MLPStopCriteria (Enum):
    """ MLP error criteria
    """
    STOP_NONE = 0
    STOP_STD  = 1
    STOP_MAX  = 2


class MLPErrorFuncs (Enum):
    """ MLP error functions
    """
    ERR_NONE          = 0
    ERR_CROSS_ENTROPY = 1
    ERR_SQUARED       = 2


class MLPRegions (Enum):
    """ SDRAM regions used by MLP cores
    """
    SYSTEM        =  0
    NETWORK       =  1
    CORE          =  2
    INPUTS        =  3
    TARGETS       =  4
    EXAMPLE_SET   =  5
    EXAMPLES      =  6
    EVENTS        =  7
    WEIGHTS       =  8
    ROUTING       =  9
    STAGE         = 10
    REC_INFO      = 11


class MLPVarSizeRecordings (Enum):
    """ t core recording channels
        with variable per-stage data size
    """
    OUTPUTS = 0


class MLPConstSizeRecordings (Enum):
    """ t core recording channels
        with constant per-stage data size
    """
    TEST_RESULTS = len (MLPVarSizeRecordings)


class MLPExtraRecordings (Enum):
    """ additional recording channels
        for first output t core
    """
    TICK_DATA = len (MLPVarSizeRecordings) + len (MLPConstSizeRecordings)
