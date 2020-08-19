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


class InputVertex(
        SimulatorVertex,
        MachineDataSpecableVertex,
        AbstractProvidesNKeysForPartition,
        AbstractRewritesDataSpecification
        ):

    """ A vertex to implement a PDP2 input core
        that applies unit input functions 
    """

    def __init__(self,
                 network,
                 group
                 ):

        super(InputVertex, self).__init__(
            label = "i_core{}".format (group.id),
            binary_name = "input.aplx",
            constraints = None)

        self._stage = 0

        # application-level data
        self._network = network
        self._group   = group
        self._set_cfg = network._ex_set.set_config
        self._ex_cfg  = network._ex_set.example_config
        self._ev_cfg  = network._ex_set.event_config

        # application parameters
        self._in_integr_dt = 1.0 / network.ticks_per_int

        # forward and backprop link partition names
        self._fwd_link = "fwd_i{}".format (self.group.id)
        self._bkp_link = []
        for p in range (self._group.partitions):
            self._bkp_link.append ("bkp_i{}_{}".format (self.group.id, p))

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

        # list of event configurations
        self._N_EVENTS_BYTES = \
            len (self._ev_cfg) * len (self._ev_cfg[0])

        # list of group inputs (empty if not an INPUT group)
        self._N_INPUTS_BYTES = \
            len (self._group.inputs) * _data_int.size

        # keys are integers
        # i cores require a different key for every group partition
        self._N_KEYS_BYTES = _data_int.size * \
            (MLPConstants.NUM_KEYS_REQ + self.group.partitions)

        # stage configuration structure
        self._N_STAGE_CONFIGURATION_BYTES = len (self._network.stage_config)

        # reserve SDRAM space used to store historic data
        self._NET_HISTORY_BYTES = (MLPConstants.LONG_NET_SIZE // 8) * \
            self.group.units * self._network.global_max_ticks


        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLE_SET_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_EVENTS_BYTES + \
            self._N_INPUTS_BYTES + \
            self._N_KEYS_BYTES + \
            self._N_STAGE_CONFIGURATION_BYTES + \
            self._NET_HISTORY_BYTES
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
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) i_conf in mlp_types.h:

            typedef struct i_conf
            {
              uchar         output_grp;
              uchar         input_grp;
              uint          num_units;
              uint          partitions;
              uint          num_in_procs;
              uint          procs_list[SPINN_NUM_IN_PROCS];
              uchar         in_integr_en;
              fpreal        in_integr_dt;
              fpreal        soft_clamp_strength;
              net_t         initNets;
              activation_t  initOutput;
            } i_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # integration dt is an MLP fixed-point fpreal
        in_integr_dt = int (self._in_integr_dt * (1 << MLPConstants.FPREAL_SHIFT))

        # soft_clamp_strength is an MLP fixed-point fpreal
        soft_clamp_strength = int (self.group.soft_clamp_strength *\
                           (1 << MLPConstants.FPREAL_SHIFT))

        # init output is an MLP fixed-point activation_t
        init_output = int (self.group.init_output *\
                           (1 << MLPConstants.ACTIV_SHIFT))

        return struct.pack ("<2B2x5IB3x4i",
                            self.group.output_grp,
                            self.group.input_grp,
                            self.group.units,
                            self.group.partitions,
                            self.group.num_in_procs,
                            self.group.in_procs_list[0].value,
                            self.group.in_procs_list[1].value,
                            self.group.in_integr_en,
                            in_integr_dt,
                            soft_clamp_strength,
                            self.group.init_net,
                            init_output
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

        # Reserve and write the events region
        spec.reserve_memory_region (MLPRegions.EVENTS.value,
                                    self._N_EVENTS_BYTES)

        spec.switch_write_focus (MLPRegions.EVENTS.value)

        # write the event configurations into spec
        for ev in self._ev_cfg:
            for c in ev:
                spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the input data region (if INPUT group)
        if self._N_INPUTS_BYTES != 0:
            spec.reserve_memory_region (MLPRegions.INPUTS.value,
                                        self._N_INPUTS_BYTES)

            spec.switch_write_focus (MLPRegions.INPUTS.value)

            # write inputs to spec
            for _i in self._group.inputs:
                # inputs are MLP fixed-point activation_t
                if (_i is None) or (_i == float ('nan')):
                    _inp = MLPConstants.ACTIV_NaN
                else:
                    _inp = int (_i * (1 << MLPConstants.ACTIV_SHIFT))
                spec.write_value (_inp, data_type = DataType.UINT32)

        # Reserve and write the routing region
        spec.reserve_memory_region (MLPRegions.ROUTING.value,
                                    self._N_KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys: fwd
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fwd_link), data_type = DataType.UINT32)

        # write link keys: bkp (padding - keys written below)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: fds (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: stp (padding),
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: lds (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: bkpi
        for p in range (self.group.partitions):
            spec.write_value (routing_info.get_first_key_from_pre_vertex (
                self, self.bkp_link[p]), data_type = DataType.UINT32)

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
