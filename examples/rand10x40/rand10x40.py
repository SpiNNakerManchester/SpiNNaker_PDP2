from spinn_pdp2.mlp_network import MLPNetwork
from spinn_pdp2.mlp_types   import MLPNetworkTypes, MLPGroupTypes, MLPUpdateFuncs
from spinn_pdp2.mlp_types   import MLPInputProcs

#-----------------------------------------------------------
# rand10x40
#
# implementation of Lens example rand10x40
#
# (https://github.com/crcox/lens/blob/master/Examples/rand10x40.in)
#
#-----------------------------------------------------------

# instantiate the MLP network
rand10x40 = MLPNetwork (net_type = MLPNetworkTypes.CONTINUOUS,
                        intervals = 4,
                        ticks_per_interval = 5
                        )

# instantiate network groups (layers)
Input  = rand10x40.group (units = 10,
                          group_type = [MLPGroupTypes.INPUT],
                          label = "Input"
                          )
Hidden = rand10x40.group (units = 50,
                          input_funcs = [MLPInputProcs.IN_INTEGR],
                          label = "Hidden"
                          )
Output = rand10x40.group (units = 10,
                          group_type = [MLPGroupTypes.OUTPUT],
                          label = "Output"
                          )

# instantiate network links
rand10x40.link (Input,  Hidden)
rand10x40.link (Hidden, Output)

# instantiate network example set
set1 = rand10x40.example_set (label = "set1")

# read Lens-style examples file
set1.read_Lens_examples_file ("rand10x40.ex")

# set example set parameters
set1.set (grace_time = 1.0,
          min_time   = 1.0,
          max_time   = 4.0
          )

# set network parameters
rand10x40.set (num_updates = 300,
               train_group_crit = 0.2
               )

# read initial weights from Lens-generated file
rand10x40.read_Lens_weights_file (
    "rand10x40_weights.txt")

# test the network for 20 examples
rand10x40.test (num_examples = 20)

# train the network for the default number of updates
rand10x40.train (update_function = MLPUpdateFuncs.UPD_STEEPEST)

# test the network again for 20 example
rand10x40.test (num_examples = 20)

# close the application
rand10x40.end ()
