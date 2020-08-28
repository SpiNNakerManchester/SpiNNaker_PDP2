from spinn_pdp2.mlp_types import MLPGroupTypes, MLPConstants
from spinn_pdp2.mlp_types import MLPInputProcs, MLPOutputProcs
from spinn_pdp2.mlp_types import MLPStopCriteria, MLPErrorFuncs

class MLPGroup():
    """ an MLP group
    """

    def __init__(self,
                 gid,
                 units        = None,
                 gtype        = [MLPGroupTypes.HIDDEN],
                 input_funcs  = None,
                 output_funcs = None,
                 write_blk    = None,
                 is_first_out = None,
                 label        = None,
                 VERBOSE      = False
                 ):
        """
        """
        self.id           = gid
        self.units        = units
        self.type         = gtype
        self.write_blk    = write_blk
        self.is_first_out = is_first_out
        self.label        = label

        # number of subgroups required for this group
        self.subgroups = (self.units + MLPConstants.MAX_SUBGROUP_UNITS - 1)\
            // MLPConstants.MAX_SUBGROUP_UNITS

        if VERBOSE:
            s = '' if self.subgroups == 1 else 's'
            print (f"creating group {self.label} with "
                   f"{self.subgroups} subgroup{s}"
                   )

        # number of units per subgroup
        self.subunits = [MLPConstants.MAX_SUBGROUP_UNITS] * (self.subgroups - 1)
        self.subunits.append (self.units - sum (self.subunits))

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
        self.s_vertex   = []
        self.i_vertex   = []
        self.t_vertex   = []

        # group function parameters
        self.output_grp = (MLPGroupTypes.OUTPUT in self.type)
        self.input_grp  = (MLPGroupTypes.INPUT in self.type)

        # weight-related parameters
        self.learning_rate = None
        self.weight_decay = None
        self.momentum = None

        # input function parameters
        self.hard_clamp_en = 0
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
            # output groups have a default output integrator unless there is an input integrator
            if (MLPGroupTypes.OUTPUT in self.type and self.in_integr_en == 0):
                self.out_integr_en  = 1
                self.num_out_procs  = MLPConstants.DEF_OUT_PROCS
                self.out_procs_list = [MLPOutputProcs.OUT_LOGISTIC,\
                                       MLPOutputProcs.OUT_INTEGR,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE,\
                                       MLPOutputProcs.OUT_NONE]
            # an input integrator removes the default output integrator from an output group
            # other groups have no integrator by default
            else:
                self.out_integr_en = 0
                self.num_out_procs = MLPConstants.DEF_OUT_PROCS - 1
                self.out_procs_list = [MLPOutputProcs.OUT_LOGISTIC,\
                                       MLPOutputProcs.OUT_NONE,\
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
        self.write_out = (MLPGroupTypes.OUTPUT in self.type)

        # group type modifies default values
        if (MLPGroupTypes.BIAS in self.type):
            self.out_integr_en      = 0
            self.num_out_procs      = 1
            self.out_procs_list [0] = MLPOutputProcs.OUT_BIAS
            self.out_procs_list [1] = MLPOutputProcs.OUT_NONE
            self.init_output        = MLPConstants.BIAS_INIT_OUT

        else:
            if (MLPGroupTypes.INPUT in self.type and MLPGroupTypes.OUTPUT not in self.type):
                self.hard_clamp_en      = 1
                self.out_integr_en      = 0
                self.num_out_procs      = 1
                self.out_procs_list [0] = MLPOutputProcs.OUT_HARD_CLAMP
                self.out_procs_list [1] = MLPOutputProcs.OUT_NONE

            if (MLPGroupTypes.OUTPUT in self.type):
                self.group_criterion    = MLPConstants.DEF_GRP_CRIT
                self.criterion_function = MLPStopCriteria.STOP_STD
                self.error_function     = MLPErrorFuncs.ERR_CROSS_ENTROPY
