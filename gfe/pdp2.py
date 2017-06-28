import spinnaker_graph_front_end as g
from gfe import binarys
from gfe.input_vertex import InputVertex
from pacman.model.graphs.machine.machine_edge import MachineEdge

g.setup(model_binary_module=binarys)

input_vertex = InputVertex(True, True)
input_vertex_2 = InputVertex(False, False)
g.add_machine_vertex_instance(input_vertex)
g.add_machine_vertex_instance(input_vertex_2)

edge = MachineEdge(input_vertex, input_vertex_2)
edge_2 = MachineEdge(input_vertex_2, input_vertex)

g.add_machine_edge_instance(edge, "ForwardPass")
g.add_machine_edge_instance(edge, "BackwardPass")

# g.run_until_finished()
g.run(None)

txrx = g.transceiver()
placements = g.placements()
placement = placements.get_placement_of_vertex(input_vertex)
input_vertex.get_data(txrx, placement)
