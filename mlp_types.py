from enum import Enum


class MLPConstants ():
    """ MLP network constants
    """
    DEF_LEARN_RT  = 0x0ccc

    DEF_INIT_NET  = 0
    DEF_INIT_OUT  = 0x4000

    BIAS_INIT_OUT = 0x7fff

    MAX_IN_PROCS  = 2
    DEF_IN_PROCS  = 0

    MAX_OUT_PROCS = 5
    DEF_OUT_PROCS = 2

    DEF_INTEGR_DT = 0x00003333
    DEF_SOFT_CLMP = 0x00008000
    DEF_WEAK_CLMP = 0

    DEF_GRP_CRIT  = 0


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
