# Copyright (c) 2022 The University of Manchester
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
# simple_past_tense
#
# a simple model that learns to associate the present tense of English verbs with
# the appropriate past tense form e.g. aching-ached
#
#-----------------------------------------------------------

# instantiate the MLP network
simple_past_tense = MLPNetwork (net_type = MLPNetworkTypes.CONTINUOUS,
                        intervals = 4,
                        ticks_per_interval = 5
                        )

# instantiate network groups (layers)
Input  = simple_past_tense.group (units = 240,
                          group_type = [MLPGroupTypes.INPUT],
                          label = "Input"
                          )
Hidden = simple_past_tense.group (units = 500,
                          input_funcs = [MLPInputProcs.IN_INTEGR],
                          label = "Hidden"
                          )
Output = simple_past_tense.group (units = 240,
                          group_type = [MLPGroupTypes.OUTPUT],
                          label = "Output"
                          )

# instantiate network links
simple_past_tense.link (Input,  Hidden)
simple_past_tense.link (Hidden, Output)

# instantiate network example set
set1 = simple_past_tense.example_set (label = "set1")

# read Lens-style examples file
set1.read_Lens_examples_file ("simple_past_tense.ex")

# set example set parameters
set1.set (grace_time = 1.0,
          min_time   = 1.0,
          max_time   = 4.0
          )

# set network parameters
simple_past_tense.set (num_presentations = 10,
               test_group_crit = 0.2
               )

# set the name for the output files
model = "simple_past_tense"

# set recording options
simple_past_tense.recording_options (rec_test_results = True,
                             results_file = "%s_results.out" % model,
                             rec_outputs = True,
                             rec_example_last_tick_only = True
                             )

# read initial weights from Lens-generated file
simple_past_tense.read_Lens_weights_file (
    "simple_past_tense_weights.txt")

# do an intial test of the network
simple_past_tense.test ()

# generate Lens-style output file
simple_past_tense.write_Lens_output_file ("%s_test.out" % model)

# train using steepest descent
simple_past_tense.train (update_function = MLPUpdateFuncs.UPD_STEEPEST)

# generate Lens-style output file
simple_past_tense.write_Lens_output_file ("%s_train.out" % model)

# do a final test of the network
simple_past_tense.test ()

# generate Lens-style output file
simple_past_tense.write_Lens_output_file ("%s_train_test.out" % model)

# close the application
simple_past_tense.end ()
