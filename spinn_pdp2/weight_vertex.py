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

from spinnaker_graph_front_end.utilities import SimulatorVertex
from spinnaker_graph_front_end.utilities.data_utils \
    import generate_steps_system_data_region

from spinn_pdp2.mlp_types import MLPRegions, MLPConstants


class WeightVertex(
        SimulatorVertex,
        MachineDataSpecableVertex,
        AbstractRewritesDataSpecification
        ):

    """ A vertex to implement a PDP2 weight core
        that computes partial weight/input products
    """

    def __init__(self,
                 network,
                 group,
                 subgroup,
                 from_group,
                 from_subgroup
                 ):

        self._network       = network
        self._group         = group
        self._from_group    = from_group
        self._subgroup      = subgroup
        self._from_subgroup = from_subgroup

        super(WeightVertex, self).__init__(
            label = (f"w_core{self.group.id}/{self.subgroup}"
                     f"_{self.from_group.id}/{self.from_subgroup}"),
            binary_name = "weight.aplx")

        self._stage = 0

        # application-level data
        self._set_cfg = self.network.ex_set.set_config
        self._ex_cfg  = self.network.ex_set.example_config

        # application parameters
        if len (self.group.weights[self.from_group]):
            if self.group.learning_rate is not None:
                self._learning_rate = self.group.learning_rate
            elif network.learning_rate is not None:
                self._learning_rate = network.learning_rate
            else:
                self._learning_rate = MLPConstants.DEF_LEARNING_RATE

            if self.group.weight_decay is not None:
                self._weight_decay = self.group.weight_decay
            elif network.weight_decay is not None:
                self._weight_decay = network.weight_decay
            else:
                self._weight_decay = MLPConstants.DEF_WEIGHT_DECAY

            if self.group.momentum is not None:
                self._momentum = self.group.momentum
            elif network.momentum is not None:
                self._momentum = network.momentum
            else:
                self._momentum = MLPConstants.DEF_MOMENTUM
        else:
            self._learning_rate = 0
            self._weight_decay = 0
            self._momentum = 0

        # forward, backprop and link delta summation link names
        self._fwd_link = (f"fwd_w{self.group.id}/{self.subgroup}"
                          f"_{self.from_group.id}/{self.from_subgroup}")

        self._bkp_link = (f"bkp_w{self.group.id}/{self.subgroup}"
                          f"_{self.from_group.id}/{self.from_subgroup}")

        self._lds_link = (f"lds_w{self.group.id}/{self.subgroup}"
                          f"_{self.from_group.id}/{self.from_subgroup}")

        self._fsg_link = (f"fsg_w{self.group.id}/{self.subgroup}"
                          f"_{self.from_group.id}/{self.from_subgroup}")

        # weight core-specific parameters
        # weight matrix parameters
        self._num_rows = self.from_group.subunits[self.from_subgroup]
        self._num_cols = self.group.subunits[self.subgroup]

        # configuration and data sizes
        # network configuration structure
        self._NETWORK_CONFIGURATION_BYTES = len (self.network.network_config)

        # core configuration structure
        self._CORE_CONFIGURATION_BYTES = len (self.config)

        # set configuration structure
        self._EXAMPLE_SET_BYTES = len (self._set_cfg)

        # list of example configurations
        self._EXAMPLES_BYTES = len (self._ex_cfg) * len (self._ex_cfg[0])

        # each weight is an integer
        self._WEIGHTS_BYTES = (self._num_rows *
                               self._num_cols * DataType.INT32.size)

        # list of routing keys
        self._KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * DataType.INT32.size

        # stage configuration structure
        self._STAGE_CONFIGURATION_BYTES = len (self.network.stage_config)

        # reserve SDRAM space used to store historic data
        self._OUTPUT_HISTORY_BYTES = ((MLPConstants.ACTIV_SIZE // 8) *
            self.group.units * self.network.global_max_ticks)

        self._sdram_usage = (
            self._NETWORK_CONFIGURATION_BYTES +
            self._CORE_CONFIGURATION_BYTES +
            self._EXAMPLE_SET_BYTES +
            self._EXAMPLES_BYTES +
            self._WEIGHTS_BYTES +
            self._KEYS_BYTES +
            self._STAGE_CONFIGURATION_BYTES +
            self._OUTPUT_HISTORY_BYTES
        )

    def cast_float_to_weight (self,
                              wt_float
                              ):
        """ casts a float into an MLP fixed-point weight_t
        """
        # round weight
        if wt_float >= 0:
            wt_float = wt_float + MLPConstants.WF_EPS / 2.0
        else:
            wt_float = wt_float - MLPConstants.WF_EPS / 2.0

        # saturate weight
        if wt_float >= MLPConstants.WF_MAX:
            wtemp = MLPConstants.WF_MAX;
            print (f"warning: input weight >= {MLPConstants.WF_MAX}")
        elif wt_float <= MLPConstants.WF_MIN:
            wtemp = MLPConstants.WF_MIN;
            print (f"warning: input weight <= {MLPConstants.WF_MIN}")
        else:
            wtemp = wt_float

        # return an MLP fixed-point weight_t
        return (int (wtemp * (1 << MLPConstants.WEIGHT_SHIFT)))

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
    def from_group (self):
        return self._from_group

    @property
    def from_subgroup (self):
        return self._from_subgroup

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
    def fsg_link (self):
        return self._fsg_link

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) w_conf in mlp_types.h:

            typedef struct w_conf
            {
              uint           num_rows;
              uint           num_cols;
              activation_t   initOutput;
              short_fpreal_t learningRate;
              short_fpreal_t weightDecay;
              short_fpreal_t momentum;
            } w_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # init output is an MLP fixed-point activation_t
        init_output = int (self.from_group.init_output *\
                           (1 << MLPConstants.ACTIV_SHIFT))

        # learning_rate is an MLP short fixed-point fpreal
        learning_rate = int (self._learning_rate *\
                              (1 << MLPConstants.SHORT_FPREAL_SHIFT))

        # weight_decay is an MLP short fixed-point fpreal
        weight_decay = int (self._weight_decay *\
                              (1 << MLPConstants.SHORT_FPREAL_SHIFT))

        # momentum is an MLP short fixed-point fpreal
        momentum = int (self._momentum *\
                              (1 << MLPConstants.SHORT_FPREAL_SHIFT))

        return struct.pack ("<2Ii3h2x",
                            self._num_rows,
                            self._num_cols,
                            init_output,
                            learning_rate,
                            weight_decay,
                            momentum
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

        # Reserve and write the weights region
        spec.reserve_memory_region (MLPRegions.WEIGHTS.value,
                                    self._WEIGHTS_BYTES)

        spec.switch_write_focus (MLPRegions.WEIGHTS.value)

        # weight matrix is kept in column-major order
        # and has to be written out in row-major order
        wts = self.group.weights[self.from_group]
        rows_per_col = self.from_group.units
        rb = self.from_subgroup * MLPConstants.MAX_SUBGROUP_UNITS
        cb = self.subgroup * MLPConstants.MAX_SUBGROUP_UNITS
        if len (wts):
            for r in range (self._num_rows):
                for c in range (self._num_cols):
                    wt = self.cast_float_to_weight (
                        wts[(cb + c) * rows_per_col + (rb + r)])
                    spec.write_value (wt, data_type = DataType.INT32)
        else:
            for _ in range (self._num_rows * self._num_cols):
                spec.write_value (0, data_type = DataType.INT32)

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
        spec.write_value(key, data_type=DataType.UINT32)

        # write link keys: bps (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: stp (padding)
        spec.write_value (0, data_type = DataType.UINT32)

        # write link keys: lds
        key = routing_info.get_key_from(
            self, self.lds_link)
        spec.write_value(key, data_type=DataType.UINT32)

        # write link keys: fsg
        key = routing_info.get_key_from(
            self, self.fsg_link)
        spec.write_value(key, data_type=DataType.UINT32)

        # Reserve and write the stage configuration region
        spec.reserve_memory_region (MLPRegions.STAGE.value,
                                    self._STAGE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.STAGE.value)

        # write the stage configuration into spec
        for c in self.network.stage_config:
            spec.write_value (c, data_type = DataType.UINT8)

        spec.end_specification ()


    @overrides(AbstractRewritesDataSpecification.regenerate_data_specification)
    def regenerate_data_specification(
            self, spec: DataSpecificationReloader, placement: Placement):
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
    def set_reload_required(self, new_value: bool):
        """
            TODO: not really sure what this method is used for!
        """
        # prepare for next stage
        self._stage += 1
