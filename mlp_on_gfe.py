import logging

import spinnaker_graph_front_end as g

from input_vertex     import InputVertex
from sum_vertex       import SumVertex
from threshold_vertex import ThresholdVertex
from weight_vertex    import WeightVertex

logger = logging.getLogger (__name__)

# Set up the simulation
g.setup ()

#------v- group 2 -v------
# instantiate group 2 cores and place them appropriately
wv2_0 = WeightVertex    (group=2, chip_x=0, chip_y=0, core=1)
wv2_1 = WeightVertex    (group=2, chip_x=0, chip_y=0, core=2)
wv2_2 = WeightVertex    (group=2, chip_x=0, chip_y=0, core=3)
wv2_3 = WeightVertex    (group=2, chip_x=0, chip_y=0, core=4)
sv2   = SumVertex       (group=2, chip_x=0, chip_y=0, core=5)
iv2   = InputVertex     (group=2, chip_x=0, chip_y=0, core=6)
tv2   = ThresholdVertex (group=2, chip_x=0, chip_y=0, core=7)

# add group 2 vertices to graph
g.add_machine_vertex_instance (wv2_0)
g.add_machine_vertex_instance (wv2_1)
g.add_machine_vertex_instance (wv2_2)
g.add_machine_vertex_instance (wv2_3)
g.add_machine_vertex_instance (sv2)
g.add_machine_vertex_instance (iv2)
g.add_machine_vertex_instance (tv2)
#------^- group 2 -^------


#------v- group 3 -v------
# instantiate group 3 cores and place them appropriately
wv3_0 = WeightVertex    (group=3, chip_x=0, chip_y=0, core=8)
wv3_1 = WeightVertex    (group=3, chip_x=0, chip_y=0, core=9)
wv3_2 = WeightVertex    (group=3, chip_x=0, chip_y=0, core=10)
wv3_3 = WeightVertex    (group=3, chip_x=0, chip_y=0, core=11)
sv3   = SumVertex       (group=3, chip_x=0, chip_y=0, core=12)
iv3   = InputVertex     (group=3, chip_x=0, chip_y=0, core=13)
tv3   = ThresholdVertex (group=3, chip_x=0, chip_y=0, core=14)

# add group 3 vertices to graph
g.add_machine_vertex_instance (wv3_0)
g.add_machine_vertex_instance (wv3_1)
g.add_machine_vertex_instance (wv3_2)
g.add_machine_vertex_instance (wv3_3)
g.add_machine_vertex_instance (sv3)
g.add_machine_vertex_instance (iv3)
g.add_machine_vertex_instance (tv3)
#------^- group 3 -^------


#------v- group 4 -v------
# instantiate group 4 cores and place them appropriately
wv4_0 = WeightVertex    (group=4, chip_x=0, chip_y=0, core=15)
wv4_1 = WeightVertex    (group=4, chip_x=0, chip_y=0, core=16)
wv4_2 = WeightVertex    (group=4, chip_x=0, chip_y=1, core=1)
wv4_3 = WeightVertex    (group=4, chip_x=0, chip_y=1, core=2)
sv4   = SumVertex       (group=4, chip_x=0, chip_y=1, core=3)
iv4   = InputVertex     (group=4, chip_x=0, chip_y=1, core=4)
tv4   = ThresholdVertex (group=4, chip_x=0, chip_y=1, core=5)

# add group 4 vertices to graph
g.add_machine_vertex_instance (wv4_0)
g.add_machine_vertex_instance (wv4_1)
g.add_machine_vertex_instance (wv4_2)
g.add_machine_vertex_instance (wv4_3)
g.add_machine_vertex_instance (sv4)
g.add_machine_vertex_instance (iv4)
g.add_machine_vertex_instance (tv4)
#------^- group 4 -^------


#------v- group 5 -v------
# instantiate group 5 cores and place them appropriately
wv5_0 = WeightVertex    (group=5, chip_x=0, chip_y=1, core=6)
wv5_1 = WeightVertex    (group=5, chip_x=0, chip_y=1, core=7)
wv5_2 = WeightVertex    (group=5, chip_x=0, chip_y=1, core=8)
wv5_3 = WeightVertex    (group=5, chip_x=0, chip_y=1, core=9)
sv5   = SumVertex       (group=5, chip_x=0, chip_y=1, core=10)
iv5   = InputVertex     (group=5, chip_x=0, chip_y=1, core=11)
tv5   = ThresholdVertex (group=5, chip_x=0, chip_y=1, core=12)

# add group 5 vertices to graph
g.add_machine_vertex_instance (wv5_0)
g.add_machine_vertex_instance (wv5_1)
g.add_machine_vertex_instance (wv5_2)
g.add_machine_vertex_instance (wv5_3)
g.add_machine_vertex_instance (sv5)
g.add_machine_vertex_instance (iv5)
g.add_machine_vertex_instance (tv5)
#------^- group 5 -^------


# Run the simulation for a second
g.run (None)

# close the machine
# g.stop ()
