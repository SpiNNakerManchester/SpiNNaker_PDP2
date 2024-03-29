# Copyright (c) 2015 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
          min_time = 1.0,
          max_time = 4.0
          )

# set network parameters
rand10x40.set (num_presentations = 10,
               train_group_crit = 0.2
               )

# set recording option
rand10x40.recording_options (rec_test_results = True,
                             results_file = "rand10x40_test_results.out",
                             rec_outputs = True,
                             rec_example_last_tick_only = False
                             )

# read initial weights from Lens-generated file
rand10x40.read_Lens_weights_file (
    "rand10x40_weights.txt")

# test the network for 20 examples
rand10x40.test (num_examples = 20)

# generate Lens-style output file
rand10x40.write_Lens_output_file ("rand10x40_test_20e.out")

# train the network for the default number of updates
rand10x40.train (update_function = MLPUpdateFuncs.UPD_STEEPEST)

# generate Lens-style output file
rand10x40.write_Lens_output_file ("rand10x40_train.out")

# uncomment to pause execution between stages for debugging
#rand10x40.pause ()

# test the network again for 20 example
rand10x40.test (num_examples = 20)

# generate Lens-style output file
rand10x40.write_Lens_output_file ("rand10x40_train_test_20e.out")

# close the application
rand10x40.end ()
