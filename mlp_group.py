from mlp_types import MLPGroupTypes, MLPConstants
from mlp_types import MLPInputProcs, MLPOutputProcs
from mlp_types import MLPStopCriteria, MLPErrorFuncs

class MLPGroup():
    """ an MLP group
    """

    def __init__(self,
                 gid,
                 units        = None,
                 gtype        = MLPGroupTypes.HIDDEN,
                 input_funcs  = None,
                 output_funcs = None,
                 write_blk    = None,
                 is_first_out = None,
                 label        = None
                 ):
        """
        """
        self.id           = gid
        self.units        = units
        self.type         = gtype
        self.write_blk    = write_blk
        self.is_first_out = is_first_out
        self.label        = label

        print "creating group {}".format (self.label)

        # keep track of associated incoming links
        self.links_from = []

        # group has no initial weights
        self.weights = dict ()

        # group has no inputs
        self.inputs = []

        # group has no targets
        self.targets = []

        # keep track of associated vertices
        self.w_vertices = []
        self.s_vertex   = None
        self.i_vertex   = None
        self.t_vertex   = None

        # group function parameters
        self.output_grp = (self.type == MLPGroupTypes.OUTPUT)
        self.input_grp  = (self.type == MLPGroupTypes.INPUT)

        # weight-related parameters
        self.learning_rate = MLPConstants.DEF_LEARNING_RATE

        # input function parameters
        self.soft_clamp_strength = MLPConstants.DEF_SOFT_CLMP

        if input_funcs is None:
            self.in_integr_en  = 0
            self.num_in_procs  = 0
            self.in_procs_list = [MLPInputProcs.IN_NONE,\
                                  MLPInputProcs.IN_NONE]
        else:
            self.num_in_procs  = len (input_funcs)
            self.in_procs_list = input_funcs
            if len (self.in_procs_list) < MLPConstants.MAX_IN_PROCS:
                self.in_procs_list.append (MLPInputProcs.IN_NONE)

            # check if input integrator requested
            if MLPInputProcs.IN_INTEGR in input_funcs:
                self.in_integr_en = 1
            else:
                self.in_integr_en = 0

        # output function parameters
        self.weak_clamp_strength = MLPConstants.DEF_WEAK_CLMP

        if output_funcs is None:
            # an input integrator removes the default output integrator
            if (self.in_integr_en == 1):
                self.out_integr_en = 0
                self.num_out_procs = MLPConstants.DEF_OUT_PROCS - 1
                self.out_procs_list = [MLPOutputProcs.OUT_LOGISTIC,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE]
            else:
                self.out_integr_en  = 1
                self.num_out_procs  = MLPConstants.DEF_OUT_PROCS
                self.out_procs_list = [MLPOutputProcs.OUT_LOGISTIC,\
                                       MLPOutputProcs.OUT_INTEGR,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE]
        else:
            self.num_out_procs  = len (output_funcs)
            self.out_procs_list = output_funcs
            if len (self.out_procs_list) < MLPConstants.MAX_OUT_PROCS:
                for _ in range (len (self.out_procs_list),\
                                MLPConstants.MAX_OUT_PROCS):
                    self.out_procs_list.append (MLPOutputProcs.OUT_NONE)

            # check if output integrator requested
            if MLPOutputProcs.OUT_INTEGR in output_funcs:
                self.out_integr_en = 1
            else:
                self.out_integr_en = 0

        # network convergence parameters
        self.train_group_crit   = None
        self.test_group_crit    = None
        self.criterion_function = MLPStopCriteria.STOP_NONE
        self.error_function     = MLPErrorFuncs.ERR_NONE

        # initialisation parameters
        self.init_net    = MLPConstants.DEF_INIT_NET
        self.init_output = MLPConstants.DEF_INIT_OUT

        # host communication parameters
        self.write_out = (self.type == MLPGroupTypes.OUTPUT)

        # group type modifies default values
        if (self.type == MLPGroupTypes.BIAS):
            self.out_integr_en      = 0
            self.num_out_procs      = 1
            self.out_procs_list [0] = MLPOutputProcs.OUT_BIAS
            self.out_procs_list [1] = MLPOutputProcs.OUT_NONE
            self.init_output        = MLPConstants.BIAS_INIT_OUT

        elif (self.type == MLPGroupTypes.INPUT):
            self.out_integr_en      = 0
            self.num_out_procs      = 1
            self.out_procs_list [0] = MLPOutputProcs.OUT_HARD_CLAMP
            self.out_procs_list [1] = MLPOutputProcs.OUT_NONE

        elif (self.type == MLPGroupTypes.OUTPUT):
            self.write_out          = 1
            self.group_criterion    = MLPConstants.DEF_GRP_CRIT
            self.criterion_function = MLPStopCriteria.STOP_STD
            self.error_function     = MLPErrorFuncs.ERR_CROSS_ENTROPY
