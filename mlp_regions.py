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
