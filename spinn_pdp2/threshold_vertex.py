import struct

from data_specification.enums.data_type import DataType

from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint
from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.resources.resource_container \
    import ResourceContainer, ConstantSDRAM
from pacman.model.resources.iptag_resource import IPtagResource

from spinn_front_end_common.abstract_models.abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models.impl \
    import MachineDataSpecableVertex

from spinn_front_end_common.utilities.constants \
    import SYSTEM_BYTES_REQUIREMENT
from spinnaker_graph_front_end.utilities import SimulatorVertex
from spinnaker_graph_front_end.utilities.data_utils \
    import generate_system_data_region

from spinn_utilities.overrides import overrides

from spinn_pdp2.mlp_types import MLPRegions, MLPConstants


class ThresholdVertex(
        SimulatorVertex,
        MachineDataSpecableVertex,
        AbstractProvidesNKeysForPartition):

    """ A vertex to implement a PDP2 threshold core
        that applies unit output and activation functions
    """

    def __init__(self,
                 network,
                 group
                 ):

        # add placement constraint if OUTPUT group
        if group.output_grp:
            constraints = [ChipAndCoreConstraint (x = 0, y = 0)]
        else:
            constraints = None

        super(ThresholdVertex, self).__init__(
            label = "t{} core".format (group.id),
            binary_name = "threshold.aplx",
            constraints = constraints)

        # application-level data
        self._network = network
        self._group   = group
        self._set_cfg = network._ex_set.set_config
        self._ex_cfg  = network._ex_set.example_config
        self._ev_cfg  = network._ex_set.event_config

        # application parameters
        self._out_integr_dt = 1.0 / network.ticks_per_int

        # choose appropriate group criterion
        if network.training:
            if self.group.train_group_crit is not None:
                self._group_criterion = self.group.train_group_crit
            elif network._train_group_crit is not None:
                self._group_criterion = network._train_group_crit
            else:
                self._group_criterion = MLPConstants.DEF_GRP_CRIT
        else:
            if self.group.test_group_crit is not None:
                self._group_criterion = self.group.test_group_crit
            elif network._test_group_crit is not None:
                self._group_criterion = network._test_group_crit
            else:
                self._group_criterion = MLPConstants.DEF_GRP_CRIT

        # check if last output group in daisy chain
        if group == network.output_chain[-1]:
            self._is_last_output_group = 1
        else:
            self._is_last_output_group = 0

        # forward, backprop and stop link partition names
        self._fwd_link = "fwd_t{}".format (self.group.id)
        self._bkp_link = "bkp_t{}".format (self.group.id)
        self._stp_link = "stp_t{}".format (self.group.id)

        # threshold core-specific parameters
        # NOTE: if all-zero w cores are optimised out these need reviewing
        self._fwd_sync_expect = len (self._group.w_vertices)
        # NOTE: not used any more, may need reviewing if re-introduced
        self._bkp_sync_expect = 0

        # reserve key space for every link
        self._n_keys = MLPConstants.KEY_SPACE_SIZE

        # configuration and data files
        # find out the size of an integer!
        _data_int = DataType.INT32

        # network configuration structure
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len (self._network.config)

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

        # list of group targets (empty if not an OUTPUT group)
        self._N_TARGETS_BYTES = \
            len (self._group.targets) * _data_int.size

        # keys are integers
        self._N_KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * _data_int.size

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLE_SET_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_EVENTS_BYTES + \
            self._N_INPUTS_BYTES + \
            self._N_TARGETS_BYTES + \
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
    def stp_link (self):
        return self._stp_link

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) t_conf in mlp_types.h:

            typedef struct t_conf
            {
              uchar         output_grp;
              uchar         input_grp;
              uint          num_units;        # reserve key space for every link

              scoreboard_t  fwd_sync_expect;
              scoreboard_t  bkp_sync_expect;
              uchar         write_out;
              uint          write_blk;
              uchar         out_integr_en;
              fpreal        out_integr_dt;
              uint          num_out_procs;
              uint          procs_list[SPINN_NUM_OUT_PROCS];
              fpreal        weak_clamp_strength;
              activation_t  initOutput;
              error_t       group_criterion;
              uchar         criterion_function;
              uchar         is_first_output_group;
              uchar         is_last_output_group;
              uchar         error_function;
            } t_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # integration dt is an MLP fixed-point fpreal
        out_integr_dt = int (self._out_integr_dt *\
                              (1 << MLPConstants.FPREAL_SHIFT))

        # weak_clamp_strength is an MLP fixed-point fpreal
        weak_clamp_strength = int (self.group.weak_clamp_strength *\
                           (1 << MLPConstants.FPREAL_SHIFT))

        # init output is an MLP fixed-point activation_t
        init_output = int (self.group.init_output *\
                           (1 << MLPConstants.ACTIV_SHIFT))

        # group criterion is an MLP fixed-point error_t
        group_criterion = int (self._group_criterion *\
                                (1 << MLPConstants.ERROR_SHIFT))

        return struct.pack ("<2B2x3IB3xIB3xi6I3i4B",
                            self.group.output_grp & 0xff,
                            self.group.input_grp & 0xff,
                            self.group.units,
                            self._fwd_sync_expect,
                            self._bkp_sync_expect,
                            self.group.write_out & 0xff,
                            self.group.write_blk,
                            self.group.out_integr_en & 0xff,
                            out_integr_dt,
                            self.group.num_out_procs,
                            self.group.out_procs_list[0].value,
                            self.group.out_procs_list[1].value,
                            self.group.out_procs_list[2].value,
                            self.group.out_procs_list[3].value,
                            self.group.out_procs_list[4].value,
                            weak_clamp_strength,
                            init_output,
                            group_criterion,
                            self.group.criterion_function.value & 0xff,
                            self.group.is_first_out & 0xff,
                            self._is_last_output_group & 0xff,
                            self.group.error_function.value & 0xff
                            )

    @property
    @overrides (MachineVertex.resources_required)
    def resources_required (self):
        resources = ResourceContainer (
            sdram = ConstantSDRAM(SYSTEM_BYTES_REQUIREMENT + self._sdram_usage),
            iptags = [IPtagResource (ip_address = "localhost",
                                    tag         = 2,
                                    port        = 17896,
                                    strip_sdp   = False)]
            )
        return resources

    @overrides (AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition (self, partition, graph_mapper):
        return self._n_keys

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation .c requirements
        generate_system_data_region(spec, MLPRegions.SYSTEM.value,
                                    self, machine_time_step, time_scale_factor)

        # Reserve and write the network configuration region
        spec.reserve_memory_region (MLPRegions.NETWORK.value,
                                    self._N_NETWORK_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.NETWORK.value)

        # write the network configuration into spec
        for c in self._network.config:
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

        # Reserve and write the target data region
        if self._N_TARGETS_BYTES != 0:
            spec.reserve_memory_region (MLPRegions.TARGETS.value,
                                        self._N_TARGETS_BYTES)

            spec.switch_write_focus (MLPRegions.TARGETS.value)

            # write targets to spec
            for _t in self._group.targets:
                # targets are MLP fixed-point activation_t
                if (_t is None) or (_t == float ('nan')):
                    _tgt = MLPConstants.ACTIV_NaN
                else:
                    _tgt = int (_t * (1 << MLPConstants.ACTIV_SHIFT))
                spec.write_value (_tgt, data_type = DataType.UINT32)

        # Reserve and write the routing region
        spec.reserve_memory_region (MLPRegions.ROUTING.value,
                                    self._N_KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys: fwd, bkp, fds (padding), stp, and lds (padding)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fwd_link), data_type = DataType.UINT32)

        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.bkp_link), data_type = DataType.UINT32)

        spec.write_value (0, data_type = DataType.UINT32)

        # stop key for OUTPUT groups only
        if self.group.output_grp:
            spec.write_value (routing_info.get_first_key_from_pre_vertex (
                self, self.stp_link), data_type = DataType.UINT32)
        else:
            spec.write_value (0, data_type = DataType.UINT32)

        spec.write_value (0, data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
