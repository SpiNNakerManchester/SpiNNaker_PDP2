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

import spinnaker_graph_front_end as gfe

from pacman.model.graphs.machine import MachineEdge
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


class SumVertex(
        SimulatorVertex,
        MachineDataSpecableVertex,
        AbstractRewritesDataSpecification
        ):

    """ A vertex to implement an PDP2 sum core
        that aggregates partial weight/input products
    """

    def __init__(self,
                 network,
                 group,
                 subgroup,
                 idx = 0
                 ):

        self._network  = network
        self._group    = group
        self._subgroup = subgroup
        self._idx      = idx

        # is this the root of a SumVertex tree?
        self._is_tree_root = idx == 0

        super(SumVertex, self).__init__(
            label = f"s_core{self.group.id}/{self.subgroup}/{self.idx}",
            binary_name = "sum.aplx")

        self._stage = 0

        # application-level data
        self._set_cfg = self.network.ex_set.set_config
        self._ex_cfg  = self.network.ex_set.example_config

        # forward, backprop, link delta summation and sync link names
        self._fwd_link = f"fwd_s{self.group.id}/{self.subgroup}/{self.idx}"
        self._bkp_link = f"bkp_s{self.group.id}/{self.subgroup}/{self.idx}"
        self._lds_link = f"lds_s{self.group.id}/{self.subgroup}/{self.idx}"
        self._fsg_link = f"fsg_s{self.group.id}/{self.subgroup}/{self.idx}"

        # sum core-specific parameters
        # NOTE: if all-zero w cores are optimised out these need reviewing
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

        # list of routing keys
        self._KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * (DataType.INT32).size

        # stage configuration structure
        self._STAGE_CONFIGURATION_BYTES = len (self.network.stage_config)

        self._sdram_usage = (
            self._NETWORK_CONFIGURATION_BYTES +
            self._CORE_CONFIGURATION_BYTES +
            self._EXAMPLE_SET_BYTES +
            self._EXAMPLES_BYTES +
            self._KEYS_BYTES +
            self._STAGE_CONFIGURATION_BYTES
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
    def units (self):
        return self._units

    @property
    def idx (self):
        return self._idx

    @property
    def is_tree_root (self):
        return self._is_tree_root

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
            (C struct) s_conf in mlp_types.h:

            typedef struct s_conf
            {
              uint         num_units;
              scoreboard_t fwd_expected;
              scoreboard_t bkp_expected;
              scoreboard_t lds_expected;
              scoreboard_t fsgn_expected;
              scoreboard_t bsgn_expected;
              uchar        is_first_group;
              uchar        is_tree_root;
              uchar        is_first_root;
            } s_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # check if first group in the network
        is_first_group = self.group == self.network.groups[0]

        # check if this is the root of the link delta sum s_core tree
        is_first_root = is_first_group and self.subgroup == 0 and self.is_tree_root

        # number of vertices in this SumVertex tree
        num_vrt = ((self.network.subgroups - 2) //
                   (MLPConstants.MAX_S_CORE_LINKS - 1)) + 1

        lvs = ((num_vrt - 1) * (MLPConstants.MAX_S_CORE_LINKS - 1))

        # number of expected packets
        if self.idx == (num_vrt - 1):
            # the last vertex in the tree may expect fewer packets
            #NOTE: this could be the root in a single-vertex tree
            expected = self.network.subgroups - lvs
        else:
            expected = MLPConstants.MAX_S_CORE_LINKS

        # keep track of these on a unit-by-unit basis
        fwd_expected = expected
        bkp_expected = expected

        # keep track of the total, not unit-by-unit, count of lds packets
        k = lvs // MLPConstants.MAX_S_CORE_LINKS
        if self.idx > (num_vrt - 2 - k):
            # lds packets from w cores only
            lds_expected = expected * self.units
        elif self.idx == (num_vrt - 2 - k):
            # lds packets from w cores and other s cores
            wp = lvs % MLPConstants.MAX_S_CORE_LINKS
            sp = MLPConstants.MAX_S_CORE_LINKS - wp
            lds_expected = wp * self.units + sp
        else:
            # lds packets from other s cores only
            lds_expected = MLPConstants.MAX_S_CORE_LINKS

        # first subgroup expects a partial lds from every other subgroup
        if self.is_tree_root and self.subgroup == 0:
            lds_expected += self.group.subgroups - 1

            # first group expects a partial lds from every other group
            if is_first_group:
                lds_expected += len (self.network.groups) - 1

        # backprop sync gen packets are handled by root nodes only
        if self.is_tree_root and self.subgroup == 0:
            # first subgroup expects from every other subgroup in group
            bsgn_expected = self.group.subgroups - 1

            # first group expect from every other group
            if is_first_group:
                bsgn_expected += len (self.network.groups) - 1
        else:
            bsgn_expected = 0

        # forward sync gen packets are handled by all nodes and
        # receive from w cores
        fsgn_expected = bsgn_expected + fwd_expected

        return struct.pack ("<6I3Bx",
                            self.units,
                            fwd_expected,
                            bkp_expected,
                            lds_expected,
                            fsgn_expected,
                            bsgn_expected,
                            is_first_group,
                            self.is_tree_root,
                            is_first_root
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
        key = routing_info.get_key_from (
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


#---------------------------------------------------------------------
class SumVertexTree(
        ):

    """ implements a tree of sum vertices
    """

    def __init__(self,
                 network,
                 group,
                 subgroup
                 ):

        max_links = MLPConstants.MAX_S_CORE_LINKS

        # total number of Sum Vertices needed to build the tree
        num_vrt = ((network.subgroups - 2) // (max_links - 1)) + 1

        # the root vertex is used as pre-vertex for outgoing links
        self._root = SumVertex (network, group, subgroup, 0)

        # add the root to the graph
        gfe.add_machine_vertex_instance (self.root)

        # and to the list of all tree vertices
        self._vertices = [self.root]

        # create the SumVertex tree
        free_links = max_links
        to_vrt = 0
        for vrt in range (1, num_vrt):
            # create a SumVertex
            vt = SumVertex (network, group, subgroup, vrt)

            # add it to the list of vertices
            self._vertices.append (vt)

            # add it to the graph
            gfe.add_machine_vertex_instance (vt)

            # add all SumVertex links towards the tree root
            gfe.add_machine_edge_instance (
                MachineEdge (vt, self.vertices[to_vrt]), vt.fwd_link
                )

            gfe.add_machine_edge_instance (
                MachineEdge (vt, self.vertices[to_vrt]), vt.bkp_link
                )

            gfe.add_machine_edge_instance (
                MachineEdge (vt, self.vertices[to_vrt]), vt.lds_link
                )

            gfe.add_machine_edge_instance (
                MachineEdge (vt, self.vertices[to_vrt]), vt.fsg_link
                )

            # take away one free link from vertex to_vrt
            free_links -= 1

            # if out of free links use next available vertex
            if free_links == 0:
                free_links = max_links
                to_vrt += 1

        # finally, map every pre-vertex to an available tree vertex
        self._leaf_map = {}
        for grp in network.groups:
            for sgrp in range (grp.subgroups):
                # assign available leaf vertex
                self._leaf_map[(grp.id, sgrp)] = self.vertices[to_vrt]

                # take away one free link from vertex to_vrt
                free_links -= 1

                # if out of free links use next available vertex
                if free_links == 0:
                    free_links = max_links
                    to_vrt += 1


    def leaf (self, group, subgroup):
        """ returns the leaf SumVertex to link to
            from a pre-vertex in group/subgroup

        :param group:    pre-vertex group
        :param subgroup: pre-vertex subgroup number

        :type group:    MLPGroup
        :type subgroup: integer

        :return: a SumVertex
        """
        return self._leaf_map[(group.id, subgroup)]


    @property
    def root (self):
        return self._root

    @property
    def vertices (self):
        return self._vertices
