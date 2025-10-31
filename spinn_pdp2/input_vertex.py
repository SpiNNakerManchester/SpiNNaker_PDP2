# Copyright (c) 2015 The University of Manchester
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import struct
from typing import Iterable, Optional

from spinn_machine.tags import IPTag, ReverseIPTag

from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.placements import Placement
from pacman.model.resources import ConstantSDRAM

from spinn_utilities.overrides import overrides

from spinn_front_end_common.abstract_models import \
    AbstractRewritesDataSpecification
from spinn_front_end_common.abstract_models.impl \
    import MachineDataSpecableVertex
from spinn_front_end_common.data import FecDataView
from spinn_front_end_common.interface.ds import (
    DataSpecificationGenerator, DataSpecificationReloader, DataType)
from spinn_front_end_common.utilities.constants \
    import SYSTEM_BYTES_REQUIREMENT
from spinn_front_end_common.utilities.data_utils import (
    generate_steps_system_data_region)


from spinnaker_graph_front_end.utilities import SimulatorVertex

from spinn_pdp2.mlp_types import MLPRegions, MLPConstants


class InputVertex(
        SimulatorVertex,
        MachineDataSpecableVertex,
        AbstractRewritesDataSpecification
        ):

    """ A vertex to implement a PDP2 input core
        that applies unit input functions
    """

    def __init__(self,
                 network,
                 group,
                 subgroup
                 ):

        self._network  = network
        self._group    = group
        self._subgroup = subgroup

        super(InputVertex, self).__init__(
            label = f"i_core{self.group.id}/{self.subgroup}",
            binary_name = "input.aplx")

        self._stage = 0

        # application-level data
        self._set_cfg = self.network.ex_set.set_config
        self._ex_cfg  = self.network.ex_set.example_config
        self._ev_cfg  = self.network.ex_set.event_config

        # application parameters
        self._in_integr_dt = 1.0 / self.network.ticks_per_int

        # forward and backprop link names
        self._fwd_link = f"fwd_i{self.group.id}/{self.subgroup}"
        self._bkp_link = f"bkp_i{self.group.id}/{self.subgroup}"

        # input core-specific parameters
        self._units = self.group.subunits[self.subgroup]

        # configuration and data sizes
        # network configuration structure
        self._NETWORK_CONFIGURATION_BYTES = len (self.network.network_config)

        # core configuration structure
        self._CORE_CONFIGURATION_BYTES = len (self.config)

        # set configuration structure
        self._EXAMPLE_SET_BYTES = len (self._set_cfg)

        # list of example configurations
        self._EXAMPLES_BYTES = len (self._ex_cfg) * len (self._ex_cfg[0])

        # list of event configurations
        self._EVENTS_BYTES = len (self._ev_cfg) * len (self._ev_cfg[0])

        # list of subgroup inputs (empty if not an INPUT group)
        if self.group.input_grp:
            self._INPUTS_BYTES = ((len (self.group.inputs) // self.group.units) *
                                  self._units * DataType.INT32.size)
        else:
            self._INPUTS_BYTES = 0

        # list of routing keys
        self._KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * DataType.INT32.size

        # stage configuration structure
        self._STAGE_CONFIGURATION_BYTES = len (self.network.stage_config)

        # reserve SDRAM space used to store historic data
        self._NET_HISTORY_BYTES = ((MLPConstants.LONG_NET_SIZE // 8) *
            self._units * self.network.global_max_ticks)


        self._sdram_usage = (
            self._NETWORK_CONFIGURATION_BYTES +
            self._CORE_CONFIGURATION_BYTES +
            self._EXAMPLE_SET_BYTES +
            self._EXAMPLES_BYTES +
            self._EVENTS_BYTES +
            self._INPUTS_BYTES +
            self._KEYS_BYTES +
            self._STAGE_CONFIGURATION_BYTES +
            self._NET_HISTORY_BYTES
        )

    @property
    def network (self):
        return self._network

    @property
    def group (self):
        return self._group

    @property
    def subgroup (self):
        return self._subgroup

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
              activation_t  initOutput;
            } i_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # integration dt is an MLP fixed-point fpreal
        in_integr_dt = int (self._in_integr_dt *
                            (1 << MLPConstants.FPREAL_SHIFT))

        # soft_clamp_strength is an MLP fixed-point fpreal
        soft_clamp_strength = int (self.group.soft_clamp_strength *
                                   (1 << MLPConstants.FPREAL_SHIFT))

        # init output is an MLP fixed-point activation_t
        init_output = int (self.group.init_output *
                           (1 << MLPConstants.ACTIV_SHIFT))

        return struct.pack ("<2B2x4IB3x4i",
                            self.group.output_grp,
                            self.group.input_grp,
                            self._units,
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
    @overrides (MachineVertex.sdram_required)
    def sdram_required (self) -> ConstantSDRAM:
        return ConstantSDRAM(SYSTEM_BYTES_REQUIREMENT + self._sdram_usage)

    @overrides (MachineVertex.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition_id: str) -> int:
        return MLPConstants.KEY_SPACE_SIZE

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec: DataSpecificationGenerator, placement: Placement,
            iptags: Optional[Iterable[IPTag]],
            reverse_iptags: Optional[Iterable[ReverseIPTag]]):
        routing_info = FecDataView.get_routing_infos()

        # Generate the system data region for simulation.c requirements
        generate_steps_system_data_region(spec, MLPRegions.SYSTEM.value, self)

        # Reserve and write the network configuration region
        spec.reserve_memory_region (MLPRegions.NETWORK.value,
                                    self._NETWORK_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.NETWORK.value)

        # write the network configuration into spec
        for c in self.network.network_config:
            spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the core configuration region
        spec.reserve_memory_region (MLPRegions.CORE.value,
                                    self._CORE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.CORE.value)

        # write the core configuration into spec
        for c in self.config:
            spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the example set region
        spec.reserve_memory_region (MLPRegions.EXAMPLE_SET.value,
                                    self._EXAMPLE_SET_BYTES)

        spec.switch_write_focus (MLPRegions.EXAMPLE_SET.value)

        # write the example set configuration into spec
        for c in self._set_cfg:
            spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the examples region
        spec.reserve_memory_region (MLPRegions.EXAMPLES.value,
                                    self._EXAMPLES_BYTES)

        spec.switch_write_focus (MLPRegions.EXAMPLES.value)

        # write the example configurations into spec
        for ex in self._ex_cfg:
            for c in ex:
                spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the events region
        spec.reserve_memory_region (MLPRegions.EVENTS.value,
                                    self._EVENTS_BYTES)

        spec.switch_write_focus (MLPRegions.EVENTS.value)

        # write the event configurations into spec
        for ev in self._ev_cfg:
            for c in ev:
                spec.write_value (c, data_type = DataType.UINT8)

        # Reserve and write the input data region (if INPUT group)
        if self.group.input_grp:
            spec.reserve_memory_region (MLPRegions.INPUTS.value,
                                        self._INPUTS_BYTES)

            spec.switch_write_focus (MLPRegions.INPUTS.value)

            # write inputs to spec
            us = self.subgroup * MLPConstants.MAX_SUBGROUP_UNITS
            for _ in range (len (self.group.inputs) // self.group.units):
                for i in self.group.inputs[us : us + self._units]:
                    # inputs are fixed-point activation_t
                    #NOTE: check for absent or NaN
                    if (i is None) or (i != i):
                        inp = MLPConstants.ACTIV_NaN
                    else:
                        inp = int (i * (1 << MLPConstants.ACTIV_SHIFT))
                    spec.write_value (inp, data_type = DataType.UINT32)
                us += self.group.units

        # Reserve and write the routing region
        spec.reserve_memory_region (MLPRegions.ROUTING.value,
                                    self._KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys: fwd
        key = routing_info.get_key_from(
            self, self.fwd_link)
        spec.write_value(key, data_type=DataType.UINT32)

        # write link keys: bkp
        key = routing_info.get_key_from(
            self, self.bkp_link)
        spec.write_value (key, data_type = DataType.UINT32)

        # write link keys: bps (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: stp (padding),
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: lds (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: fsg (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # Reserve and write the stage configuration region
        spec.reserve_memory_region (MLPRegions.STAGE.value,
                                    self._STAGE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.STAGE.value)

        # write the stage configuration into spec
        for c in self.network.stage_config:
            spec.write_value (c, data_type = DataType.UINT8)

        spec.end_specification ()


    @overrides(AbstractRewritesDataSpecification.regenerate_data_specification)
    def regenerate_data_specification(self, spec: DataSpecificationReloader,
                                      placement: Placement) -> None:
        # Reserve and write the stage configuration region
        spec.reserve_memory_region (MLPRegions.STAGE.value,
                                    self._STAGE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.STAGE.value)

        # write the stage configuration into spec
        for c in self.network.stage_config:
            spec.write_value (c, data_type = DataType.UINT8)

        spec.end_specification()


    @overrides(AbstractRewritesDataSpecification.reload_required)
    def reload_required(self) -> bool:
        return True


    @overrides(AbstractRewritesDataSpecification.set_reload_required)
    def set_reload_required(self, new_value: bool) -> None:
        """
            TODO: not really sure what this method is used for!
        """
        # prepare for next stage
        self._stage += 1
