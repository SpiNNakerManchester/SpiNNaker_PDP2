import struct

from data_specification.enums.data_type import DataType

from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.resources.resource_container \
    import ResourceContainer, ConstantSDRAM

from spinn_utilities.overrides import overrides

from spinn_front_end_common.abstract_models.abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models import \
    AbstractRewritesDataSpecification
from spinn_front_end_common.abstract_models.impl \
    import MachineDataSpecableVertex
from spinn_front_end_common.utilities.constants \
    import SYSTEM_BYTES_REQUIREMENT

from spinnaker_graph_front_end.utilities import SimulatorVertex
from spinnaker_graph_front_end.utilities.data_utils \
    import generate_steps_system_data_region

from spinn_pdp2.mlp_types import MLPRegions, MLPConstants


class SumVertex(
        SimulatorVertex,
        MachineDataSpecableVertex,
        AbstractProvidesNKeysForPartition,
        AbstractRewritesDataSpecification
        ):

    """ A vertex to implement an PDP2 sum core
        that aggregates partial weight/input products
    """

    def __init__(self,
                 network,
                 group
                 ):

        super(SumVertex, self).__init__(
            label = "s_core{}".format (group.id),
            binary_name = "sum.aplx",
            constraints = None)

        self._stage = 0

        # application-level data
        self._network = network
        self._group   = group
        self._set_cfg = network._ex_set.set_config
        self._ex_cfg  = network._ex_set.example_config

        # check if first group in the network
        if self.group.id == network.groups[0].id:
            self._is_first_group = 1
        else:
            self._is_first_group = 0

        # forward, backprop, and link delta summation link partition names
        self._fwd_link = "fwd_s{}".format (self.group.id)
        self._bkp_link = "bkp_s{}".format (self.group.id)
        self._lds_link = "lds_s{}".format (self.group.id)
        self._fds_link = "fds_s{}".format (self.group.id)

        # sum core-specific parameters
        # NOTE: if all-zero w cores are optimised out these need reviewing
        self._fwd_expect  = network.partitions
        self._bkp_expect  = network.partitions
        self._ldsa_expect = network.partitions * self.group.units
        self._ldst_expect = len (network.groups) - 1

        # weight update function
        self.update_function = network._update_function

        # reserve key space for every link
        self._n_keys = MLPConstants.KEY_SPACE_SIZE

        # configuration and data files
        # find out the size of an integer!
        _data_int = DataType.INT32

        # network configuration structure
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len (self._network.network_config)

        # core configuration structure
        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        # set configuration structure
        self._N_EXAMPLE_SET_BYTES = \
            len (self._set_cfg)

        # list of example configurations
        self._N_EXAMPLES_BYTES = \
            len (self._ex_cfg) * len (self._ex_cfg[0])

        # keys are integers
        self._N_KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * _data_int.size

        # stage configuration structure
        self._N_STAGE_CONFIGURATION_BYTES = \
            len (self._network.stage_config)

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLE_SET_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_KEYS_BYTES + \
            self._N_STAGE_CONFIGURATION_BYTES
        )

    @property
    def group (self):
        return self._group

    @property
    def fwd_link (self):
        return self._fwd_link

    @property
    def bkp_link (self):
        return self._bkp_link

    @property
    def lds_link (self):
        return self._lds_link

    @property
    def fds_link (self):
        return self._fds_link

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) s_conf in mlp_types.h:

            typedef struct s_conf
            {
              uint         num_units;
              scoreboard_t fwd_expect;
              scoreboard_t bkp_expect;
              scoreboard_t ldsa_expect;
              scoreboard_t ldst_expect;
              uchar        is_first_group;
            } s_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """

        return struct.pack ("<5IB3x",
                            self.group.units,
                            self._fwd_expect,
                            self._bkp_expect,
                            self._ldsa_expect,
                            self._ldst_expect,
                            self._is_first_group
                            )

    @property
    @overrides (MachineVertex.resources_required)
    def resources_required (self):
        resources = ResourceContainer (
            sdram = ConstantSDRAM(SYSTEM_BYTES_REQUIREMENT + self._sdram_usage)
            )
        return resources


    @overrides (AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition (self, partition, graph_mapper):
        return self._n_keys


    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation.c requirements
        generate_steps_system_data_region(spec, MLPRegions.SYSTEM.value, self)

        # Reserve and write the network configuration region
        spec.reserve_memory_region (MLPRegions.NETWORK.value,
                                    self._N_NETWORK_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.NETWORK.value)

        # write the network configuration into spec
        for c in self._network.network_config:
            spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the core configuration region
        spec.reserve_memory_region (MLPRegions.CORE.value,
                                    self._N_CORE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.CORE.value)

        # write the core configuration into spec
        for c in self.config:
            spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the example set region
        spec.reserve_memory_region (MLPRegions.EXAMPLE_SET.value,
                                    self._N_EXAMPLE_SET_BYTES)

        spec.switch_write_focus (MLPRegions.EXAMPLE_SET.value)

        # write the example set configuration into spec
        for c in self._set_cfg:
            spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the examples region
        spec.reserve_memory_region (MLPRegions.EXAMPLES.value,
                                    self._N_EXAMPLES_BYTES)

        spec.switch_write_focus (MLPRegions.EXAMPLES.value)

        # write the example configurations into spec
        for ex in self._ex_cfg:
            for c in ex:
                spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the routing region
        spec.reserve_memory_region (MLPRegions.ROUTING.value,
                                    self._N_KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys: fwd
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fwd_link), data_type = DataType.UINT32)

        # write link keys: bkp
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.bkp_link), data_type = DataType.UINT32)

        # write link keys: fds
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fds_link), data_type = DataType.UINT32)

        # write link keys: stp (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: lds
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.lds_link), data_type = DataType.UINT32)

        # Reserve and write the stage configuration region
        spec.reserve_memory_region (MLPRegions.STAGE.value,
                                    self._N_STAGE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.STAGE.value)

        # write the stage configuration into spec
        for c in self._network.stage_config:
            spec.write_value (c, data_type = DataType.UINT8)

        spec.end_specification ()


    @overrides(AbstractRewritesDataSpecification.regenerate_data_specification)
    def regenerate_data_specification(self, spec, placement):
        # Reserve and write the stage configuration region
        spec.reserve_memory_region (MLPRegions.STAGE.value,
                                    self._N_STAGE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.STAGE.value)

        # write the stage configuration into spec
        for c in self._network.stage_config:
            spec.write_value (c, data_type = DataType.UINT8)

        spec.end_specification()


    @overrides(AbstractRewritesDataSpecification.requires_memory_regions_to_be_reloaded)
    def requires_memory_regions_to_be_reloaded(self):
        return True


    @overrides(AbstractRewritesDataSpecification.mark_regions_reloaded)
    def mark_regions_reloaded(self):
        """
            TODO: not really sure what this method is used for!
        """
        # prepare for next stage
        self._stage += 1
