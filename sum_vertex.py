import struct
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

from mlp_network import MLPRegions


class SumVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP input core
    """

    def __init__(self,
                 network = None,
                 group = None,
                 num_nets = None,
                 fwd_expect = None,
                 bkp_expect = None
                 ):
        """
        """

        # MLP network
        self._network = network
        self._group   = group

        MachineVertex.__init__(self, label =\
                               "s{} core".format (self._group))

        # sum core-specific parameters
        self._num_nets   = num_nets
        self._fwd_expect = fwd_expect
        self._bkp_expect = bkp_expect

        # forward and backprop link partition names
        self._fwd_link = "fwd_s{}".format (self._group)
        self._bkp_link = "bkp_s{}".format (self._group)

        self._n_keys = 65536

        # binary, configuration and data files
        self._aplxFile = "binaries/sum.aplx"
        self._examplesFile = "data/examples.dat"

        # size in bytes of the data in the regions
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len ((self._network).config)

        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        self._N_EXAMPLES_BYTES = \
            os.path.getsize (self._examplesFile) \
            if os.path.isfile (self._examplesFile) \
            else 0

        self._N_KEY_BYTES = 16

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_KEY_BYTES
        )

    @property
    def fwd_link (self):
        return self._fwd_link

    @property
    def bkp_link (self):
        return self._bkp_link

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) s_conf in mlp_types.h:

            typedef struct s_conf
            {
              uint         num_nets;
              scoreboard_t fwd_expect;
              scoreboard_t bkp_expect;
            } s_conf_t;

            pack: standard sizes, little-endian byte-order,
            explicit padding
        """
        return struct.pack("<3I",
                           self._num_nets,
                           self._fwd_expect,
                           self._bkp_expect
                           )
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
    def get_n_keys_for_partition (self, partition, graph_mapper):
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
        spec.reserve_memory_region (
            MLPRegions.CORE.value, self._N_CORE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.CORE.value)

        # write the core configuration into spec
        for c in self.config:
            spec.write_value (ord (c), data_type=DataType.UINT8)

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

        # Reserve and write the routing region
        spec.reserve_memory_region (
            MLPRegions.ROUTING.value, self._N_KEY_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys (fwd, bkp)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._fwd_link), data_type = DataType.UINT32)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._bkp_link), data_type = DataType.UINT32)
        spec.write_value (0, data_type = DataType.UINT32)
        spec.write_value (0, data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
