# Copyright (c) 2015-2021 The University of Manchester
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from spinn_pdp2.mlp_network import MLPNetwork
from spinn_pdp2.mlp_types   import MLPNetworkTypes, MLPGroupTypes
from spinn_pdp2.mlp_types   import MLPInputProcs

#-----------------------------------------------------------
# visSemPhon
#
# implementation of a simple visual-semantic-phonological network
#
# SR Welbourne, AM Woollams, J Crisp and MA Lambon Ralph,
# "The role of plasticity-related functional reorganization
# in the explanation of central dyslexias"
# COGNITIVE NEUROPSYCHOLOGY, 2011, 28 (2), 65 –108
#
#-----------------------------------------------------------

# instantiate the MLP network
visSemPhon = MLPNetwork (net_type = MLPNetworkTypes.CONTINUOUS,
                        intervals = 6,
                        ticks_per_interval = 5
                        )

hidden_size = 50

# instantiate network groups (layers)
Visual  = visSemPhon.group (units = 400,
                          group_type = [MLPGroupTypes.INPUT],
                          label = "V"
                          )
Semantic_H = visSemPhon.group (units = hidden_size,
                          input_funcs = [MLPInputProcs.IN_INTEGR],
                          label = "S2"
                          )
Phono_H = visSemPhon.group (units = hidden_size,
                          input_funcs = [MLPInputProcs.IN_INTEGR],
                          label = "P2"
                          )
Semantic = visSemPhon.group (units = 800,
                          group_type = [MLPGroupTypes.OUTPUT],
                          label = "S"
                          )
Phono = visSemPhon.group (units = 61,
                          group_type = [MLPGroupTypes.OUTPUT],
                          label = "P"
                          )

# instantiate network links
visSemPhon.link (Semantic, Semantic_H)
visSemPhon.link (Semantic_H, Semantic)

visSemPhon.link (Phono, Phono_H)
visSemPhon.link (Phono_H, Phono)

visSemPhon.link (Semantic, Phono)
visSemPhon.link (Semantic_H, Phono)
visSemPhon.link (Semantic, Phono_H)
visSemPhon.link (Semantic_H, Phono_H)

visSemPhon.link (Phono, Semantic)
visSemPhon.link (Phono_H, Semantic)
visSemPhon.link (Phono, Semantic_H)
visSemPhon.link (Phono_H, Semantic_H)

visSemPhon.link (Visual, Semantic)
visSemPhon.link (Visual, Semantic_H)

# set network parameters
visSemPhon.set (num_presentations = 10000,
    learning_rate = 0.05,
    momentum = 0.9,
    train_group_crit = 0.0,
    test_group_crit = 0.5
    )

# instantiate network example set
set1 = visSemPhon.example_set (label = "set1")

# read Lens-style examples file
set1.read_Lens_examples_file ("namtr.ex")

# set example set parameters
set1.set (grace_time = 1.0,
          min_time   = 1.0,
          max_time   = 4.0
          )

# read initial weights from Lens-generated file
visSemPhon.read_Lens_weights_file (
    "visSemPhon_weights.txt")

# train the network for 1 epoch
visSemPhon.train (num_updates = 1)

# close the application
visSemPhon.end ()
