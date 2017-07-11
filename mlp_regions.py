from enum import Enum

class MLPRegions(Enum):
    """ regions used by MLP cores
    """
    GLOBAL = 0
    CHIP = 1
    CORE = 2
    INPUTS = 3
    TARGETS = 4
    EXAMPLE_SET = 5
    EXAMPLES = 6
    EVENTS = 7
    WEIGHTS = 8
    ROUTING = 9
