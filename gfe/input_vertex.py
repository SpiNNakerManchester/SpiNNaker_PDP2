from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource
from spinn_front_end_common.utilities.utility_objs.executable_start_type\
    import ExecutableStartType
from spinn_front_end_common.abstract_models\
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.impl.machine_data_specable_vertex\
    import MachineDataSpecableVertex
from pacman.executor.injection_decorator import inject_items
from data_specification.enums.data_type import DataType
import numpy
from spinn_front_end_common.abstract_models.abstract_provides_n_keys_for_partition import AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary


class InputVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification,
        MachineDataSpecableVertex, AbstractProvidesNKeysForPartition):

    def __init__(self, is_input_group, is_output_group, input_data_file):
        MachineVertex.__init__(self)
        self._is_input_group = is_input_group
        self._input_data_file = input_data_file

    @property
    def resources_required(self):
        return ResourceContainer(sdram=SDRAMResource(12345))

    def get_binary_file_name(self):
        return "input.aplx"

    def get_binary_start_type(self):
        ExecutableStartType.SYNC

    @inject_items({
        "routing_info": "MemoryRoutingInfos"
    })
    def generate_data_specification(self, spec, placement, routing_info):
        pass

    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):
        spec.reserve_memory_region(region=0, size=10, label="MyParamters")
        spec.switch_write_focus(0)
        spec.write_value(1.234, data_type=DataType.S1615)
        spec.write_value(self._is_input_group, data_type=DataType.UINT32)
        numpy.dtype("u4,f8,u4", align=True)
        spec.write_array([0, 1, 2, 3])
        key = routing_info.get_first_key_from_pre_vertex(self, "ForwardPass")
        data = read_data_file(self._input_data_file)
        spec.reserve_memory_region(region=1, size=1000, label="Examples")
        spec.switch_write_focus(1)
        spec.write_array(data)
        spec.end_specification()

    def get_n_keys_for_partition(self, partition, graph_mapper):
        if partition == "ForwardPass":
            return 5
        if partition == "ReversePass":
            return 10
        raise Exception("I don't know {}".format(partition))

    def get_data(self, txrx, placement):
        return txrx.read_memory(placement.x, placement.y, my_recording_address, my_recording_size)
