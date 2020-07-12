from spinn_pdp2.mlp_network import MLPNetwork
from spinn_pdp2.mlp_types   import MLPNetworkTypes, MLPGroupTypes, MLPUpdateFuncs
from spinn_pdp2.mlp_types   import MLPInputProcs

#-----------------------------------------------------------
# rogers-basic
#
# implementation of a simplified Rogers network
#
# Rogers et al., "The structure and deterioration of semantic
# memory: A computational and neuropsychological investigation"
# Psychological Review, 111(1), 205–235, 2004.
#
#-----------------------------------------------------------

# instantiate the MLP network
rogers = MLPNetwork (net_type = MLPNetworkTypes.CONTINUOUS,
                        intervals = 8,
                        ticks_per_interval = 10
                        )

# instantiate network groups (layers)
Encyclo  = rogers.group (units = 18,
                          group_type = [MLPGroupTypes.INPUT, MLPGroupTypes.OUTPUT], 
                          label = "Encyclo"
                          )
Functional = rogers.group (units = 32,
                          group_type = [MLPGroupTypes.INPUT, MLPGroupTypes.OUTPUT],
                          label = "Functional"
                          )
Perceptual = rogers.group (units = 61,
                          group_type = [MLPGroupTypes.INPUT, MLPGroupTypes.OUTPUT],
                          label = "Perceptual"
                          )
Visual = rogers.group (units = 64,
                          group_type = [MLPGroupTypes.INPUT, MLPGroupTypes.OUTPUT],
                          label = "Visual"
                          )
Name = rogers.group (units = 48,
                          group_type = [MLPGroupTypes.INPUT, MLPGroupTypes.OUTPUT],
                          label = "Name"
                          )
Semantic = rogers.group (units = 64,
                          label = "Semantic"
                          )

# instantiate network links
rogers.link (Encyclo, Semantic)
rogers.link (Semantic, Encyclo)
rogers.link (Functional, Semantic)
rogers.link (Semantic, Functional)
rogers.link (Perceptual, Semantic)
rogers.link (Semantic, Perceptual)
rogers.link (Visual, Semantic)
rogers.link (Semantic, Visual)
rogers.link (Name, Semantic)
rogers.link (Semantic, Name)
rogers.link (Semantic, Semantic)

# instantiate network example set
set1 = rogers.example_set (label = "set1")

# read Lens-style examples file
set1.read_Lens_examples_file ("rogers-basic.ex")

# set network parameters
rogers.set (num_updates = 10,
              learning_rate = 0.005,
              weight_decay = 0.0002,
              momentum = 0.0,
              train_group_crit = 0.2
              )

# read initial weights from Lens-generated file
rogers.read_Lens_weights_file (
    "rogers-basic_weights.txt")

# train the network for the default number of updates
rogers.train (update_function = MLPUpdateFuncs.UPD_DOUGSMOMENTUM)

# generate Lens-style output file
rogers.write_Lens_output_file ("rogers-basic_train.out")

# test the network for the complete example set
rogers.test ()

# generate Lens-style output file
rogers.write_Lens_output_file ("rogers-basic_train_test_1e.out")

# close the application
rogers.end ()
