import numpy as np
import os

from data_specification.enums.data_type import DataType

from pacman.executor.injection_decorator import inject_items

from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.decorators.overrides import overrides
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.sdram_resource import SDRAMResource
from pacman.model.resources.cpu_cycles_per_tick_resource \
    import CPUCyclesPerTickResource

from spinn_front_end_common.utilities.utility_objs.executable_start_type \
    import ExecutableStartType
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models\
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

from mlp_regions import MLPRegions


class WeightVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP input core
    """

    def __init__(self, network=None, group = None,
                 frm_grp = None, file_x = None, file_y = None, file_c = None):
        """
        """

        MachineVertex.__init__(self, label =\
                               "w{}_{} core".format (group, frm_grp))

        # MLP network
        self._network = network

        # forward and backprop link partition names
        self._fwd_link = "fwd_w{}_{}".format (group, frm_grp)
        self._bkp_link = "bkp_w{}_{}".format (group, frm_grp)
        self._fds_link = "fds_w{}_{}".format (group, frm_grp)

        self._n_keys = 65536

        # binary, configuration and data files
        self._aplxFile = "binaries/weight.aplx"
        self._coreFile = "data/w_conf_{}_{}_{}.dat".format (file_x, file_y, file_c)
        self._examplesFile = "data/examples.dat"
        self._weightsFile = "data/weights_{}_{}_{}.dat".format (file_x, file_y, file_c)

        # size in bytes of the data in the regions
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len ((self._network).config)

        self._N_CORE_CONFIGURATION_BYTES = \
            os.path.getsize (self._coreFile) \
            if os.path.isfile (self._coreFile) \
            else 0

        self._N_EXAMPLES_BYTES = \
            os.path.getsize (self._examplesFile) \
            if os.path.isfile (self._examplesFile) \
            else 0

        self._N_WEIGHTS_BYTES = \
            os.path.getsize (self._weightsFile) \
            if os.path.isfile (self._weightsFile) \
            else 0

        self._N_KEY_BYTES = 16

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_WEIGHTS_BYTES + \
            self._N_KEY_BYTES
        )

    @property
    def fwd_link (self):
        return self._fwd_link

    @property
    def bkp_link (self):
        return self._bkp_link

    @property
    def fds_link (self):
        return self._fds_link

    @property
    @overrides (MachineVertex.resources_required)
    def resources_required (self):

        resources = ResourceContainer (
            dtcm=DTCMResource (0),
            sdram=SDRAMResource (self._sdram_usage),
            cpu_cycles=CPUCyclesPerTickResource (0),
            iptags=[], reverse_iptags=[])
        return resources

    @overrides (AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name (self):
        return self._aplxFile

    @overrides (AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type (self):
        return ExecutableStartType.SYNC

    @overrides (AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return self._n_keys

    @inject_items ({
        "routing_info": "MemoryRoutingInfos"})
    @overrides (
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments=["routing_info"])
    def generate_data_specification (
            self, spec, placement, routing_info):

        # Reserve and write the network configuration region
        spec.reserve_memory_region (
            MLPRegions.NETWORK.value,
            self._N_NETWORK_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.NETWORK.value)

        # write the network configuration into spec
        for c in (self._network).config:
            spec.write_value (ord (c), data_type=DataType.UINT8)

        # Reserve and write the core configuration region
        if os.path.isfile (self._coreFile):
            spec.reserve_memory_region (
                MLPRegions.CORE.value, self._N_CORE_CONFIGURATION_BYTES)

            spec.switch_write_focus (MLPRegions.CORE.value)

            # open the core configuration file
            core_file = open (self._coreFile, "rb")

            # read the data into a numpy array and put in spec
            pc = np.fromfile (core_file, np.uint8)
            for byte in pc:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the examples region
        if os.path.isfile (self._examplesFile):
            spec.reserve_memory_region (
                MLPRegions.EXAMPLES.value,
                self._N_EXAMPLES_BYTES)

            spec.switch_write_focus (MLPRegions.EXAMPLES.value)

            # open the examples file
            examples_file = open (self._examplesFile, "rb")

            # read the data into a numpy array and put in spec
            ex = np.fromfile (examples_file, np.uint8)
            for byte in ex:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the weights region
        if os.path.isfile (self._weightsFile):
            spec.reserve_memory_region (
                MLPRegions.WEIGHTS.value,
                self._N_WEIGHTS_BYTES)

            spec.switch_write_focus (MLPRegions.WEIGHTS.value)

            # open the weights file
            routes_file = open (self._weightsFile, "rb")

            # read the data into a numpy array and put in spec
            wc = np.fromfile (routes_file, np.uint8)
            for byte in wc:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the routing region
        spec.reserve_memory_region (
            MLPRegions.ROUTING.value, self._N_KEY_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys (fwd, bkp, fds, stp)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._fwd_link), data_type = DataType.UINT32)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._bkp_link), data_type = DataType.UINT32)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._fds_link), data_type = DataType.UINT32)
        spec.write_value (0, data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
