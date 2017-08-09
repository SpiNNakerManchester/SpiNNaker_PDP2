import struct
import numpy as np
import os

from data_specification.enums.data_type import DataType

from pacman.executor.injection_decorator import inject_items

from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.decorators.overrides import overrides
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource

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

from mlp_types import MLPRegions, MLPConstants


class InputVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP input core
    """

    def __init__(self,
                 network,
                 group
                 ):
        """
        """
        MachineVertex.__init__(self, label =\
                               "i{} core".format (group.id))

        # application-level data
        self._network = network
        self._group   = group
        self._ex_cfg  = network._ex_set.example_config
        self._ev_cfg  = network._ex_set.event_config

        # forward and backprop link partition names
        self._fwd_link = "fwd_i{}".format (self.group.id)
        self._bkp_link = "bkp_i{}".format (self.group.id)

        # input core-specific parameters
        self._in_integr_dt = 1.0 / network.ticks_per_int

        # reserve a 16-bit key space in every link
        self._n_keys = MLPConstants.KEY_SPACE_SIZE

        # binary, configuration and data files
        self._aplx_file = "binaries/input.aplx"

        # find out the size of an integer!
        _data_int=DataType.INT32
        int_size = _data_int.size

        # network configuration structure
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len (self._network.config)

        # core configuration structure
        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        # list of example configurations
        self._N_EXAMPLES_BYTES = \
            len (self._ex_cfg) * len (self._ex_cfg[0])

        # list of event configurations
        self._N_EVENTS_BYTES = \
            len (self._ev_cfg) * len (self._ev_cfg[0])

        # list of group inputs (empty if not an INPUT group)
        self._N_INPUTS_BYTES = \
            len (self._group.inputs) * int_size

        # 4 keys / keys are integers
        self._N_KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * int_size

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_EVENTS_BYTES + \
            self._N_INPUTS_BYTES + \
            self._N_KEYS_BYTES
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
              uint          num_in_procs;
              uint          procs_list[SPINN_NUM_IN_PROCS];
              uchar         in_integr_en;
              fpreal        in_integr_dt;
              fpreal        soft_clamp_strength;
              net_t         initNets;
              short_activ_t initOutput;
            } i_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # integration dt is represented in fixed-point s15.16 notation
        _in_integr_dt = int (self._in_integr_dt * (1 << MLPConstants.FPREAL_SHIFT))

        return struct.pack ("<2B2x4IB3x3ih2x",
                            self.group.output_grp,
                            self.group.input_grp,
                            self.group.units,
                            self.group.num_in_procs,
                            self.group.in_procs_list[0].value,
                            self.group.in_procs_list[1].value,
                            self.group.in_integr_en,
                            _in_integr_dt,
                            self.group.soft_clamp_strength,
                            self.group.init_net,
                            self.group.init_output & 0xffff
                            )

    @property
    @overrides (MachineVertex.resources_required)
    def resources_required (self):

        resources = ResourceContainer (
            sdram = SDRAMResource (self._sdram_usage),
            )
        return resources

    @overrides (AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name (self):
        return self._aplx_file

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
        spec.reserve_memory_region (MLPRegions.NETWORK.value,
                                    self._N_NETWORK_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.NETWORK.value)

        # write the network configuration into spec
        for c in self._network.config:
            spec.write_value (ord (c), data_type = DataType.UINT8)

        # Reserve and write the core configuration region
        spec.reserve_memory_region (MLPRegions.CORE.value,
                                    self._N_CORE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.CORE.value)

        # write the core configuration into spec
        for c in self.config:
            spec.write_value (ord (c), data_type = DataType.UINT8)

        # Reserve and write the examples region
        spec.reserve_memory_region (MLPRegions.EXAMPLES.value,
                                    self._N_EXAMPLES_BYTES)

        spec.switch_write_focus (MLPRegions.EXAMPLES.value)

        # write the example configurations into spec
        for ex in self._ex_cfg:
            for c in ex:
                spec.write_value (ord (c), data_type = DataType.UINT8)

        # Reserve and write the events region
        spec.reserve_memory_region (MLPRegions.EVENTS.value,
                                    self._N_EVENTS_BYTES)

        spec.switch_write_focus (MLPRegions.EVENTS.value)

        # write the event configurations into spec
        for ev in self._ev_cfg:
            for c in ev:
                spec.write_value (ord (c), data_type = DataType.UINT8)

        # Reserve and write the input data region (if INPUT group)
        if self._N_INPUTS_BYTES != 0:
            spec.reserve_memory_region (MLPRegions.INPUTS.value,
                                        self._N_INPUTS_BYTES)

            spec.switch_write_focus (MLPRegions.INPUTS.value)

            # write inputs to spec
            for _i in self._group.inputs:
                # inputs are represented in fix-point s16.15 notation
                _inp = int (_i * (1 << MLPConstants.ACTIV_SHIFT))
                spec.write_value (_inp, data_type = DataType.UINT32)

        # Reserve and write the routing region
        spec.reserve_memory_region (MLPRegions.ROUTING.value,
                                    self._N_KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys: fwd, bkp, padding, padding
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fwd_link), data_type = DataType.UINT32)

        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.bkp_link), data_type = DataType.UINT32)

        spec.write_value (0, data_type = DataType.UINT32)

        spec.write_value (0, data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
