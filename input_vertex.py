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

        # forward and backprop link partition names
        self._fwd_link = "fwd_i{}".format (self.group.id)
        self._bkp_link = "bkp_i{}".format (self.group.id)

        # input core-specific parameters
        self._in_integr_dt = int ((1.0 / network.ticks_per_int) *\
                                  (1 << 16))

        # reserve a 16-bit key space in every link
        self._n_keys = MLPConstants.KEY_SPACE_SIZE

        # binary, configuration and data files
        self._aplx_file     = "binaries/input.aplx"
        self._inputs_file   = "data/inputs_{}.dat".\
                                format (self.group.id + 2) #lap
        self._examples_file = "data/examples.dat"
        self._events_file   = "data/events.dat"

        # find out the size of an integer!
        _data_int=DataType.INT32
        int_size = _data_int.size

        # size in bytes of the data in the regions
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len (self._network.config)

        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        self._N_INPUTS_BYTES = \
            os.path.getsize (self._inputs_file) \
            if os.path.isfile (self._inputs_file) \
            else 0

        self._N_EXAMPLES_BYTES = \
            os.path.getsize (self._examples_file) \
            if os.path.isfile (self._examples_file) \
            else 0

        self._N_EVENTS_BYTES = \
            os.path.getsize (self._events_file) \
            if os.path.isfile (self._events_file) \
            else 0

        # 4 keys / keys are integers
        self._N_KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * int_size

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_INPUTS_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_EVENTS_BYTES + \
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
        return struct.pack ("<2B2x4IB3x3ih2x",
                            self.group.output_grp,
                            self.group.input_grp,
                            self.group.units,
                            self.group.num_in_procs,
                            self.group.in_procs_list[0].value,
                            self.group.in_procs_list[1].value,
                            self.group.in_integr_en,
                            self._in_integr_dt,
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
        spec.reserve_memory_region (
            MLPRegions.NETWORK.value,
            self._N_NETWORK_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.NETWORK.value)

        # write the network configuration into spec
        for c in self._network.config:
            spec.write_value (ord (c), data_type=DataType.UINT8)

        # Reserve and write the core configuration region
        spec.reserve_memory_region (
            MLPRegions.CORE.value, self._N_CORE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.CORE.value)

        # write the core configuration into spec
        for c in self.config:
            spec.write_value (ord (c), data_type=DataType.UINT8)


        # Reserve and write the input data region
        if os.path.isfile (self._inputs_file):
            spec.reserve_memory_region (
                MLPRegions.INPUTS.value,
                self._N_INPUTS_BYTES)

#             print "iv-{}: reading {}".format (self.group.id,
#                                               self._inputs_file
#                                               )

            spec.switch_write_focus (MLPRegions.INPUTS.value)

            # open input data file
            _if = open (self._inputs_file, "rb")

            # read the data into a numpy array and put in spec
            _ic = np.fromfile (_if, np.uint8)
            _if.close ()
            for byte in _ic:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the examples region
        if os.path.isfile (self._examples_file):
            spec.reserve_memory_region (
                MLPRegions.EXAMPLES.value,
                self._N_EXAMPLES_BYTES)

#             print "iv-{}: reading {}".format (self.group.id,
#                                               self._examples_file
#                                               )

            spec.switch_write_focus (MLPRegions.EXAMPLES.value)

            # open the examples file
            _ef = open (self._examples_file, "rb")

            # read the data into a numpy array and put in spec
            _ex = np.fromfile (_ef, np.uint8)
            _ef.close ()
            for byte in _ex:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the events region
        if os.path.isfile (self._events_file):
            spec.reserve_memory_region (
                MLPRegions.EVENTS.value,
                self._N_EVENTS_BYTES)

#             print "iv-{}: reading {}".format (self.group.id,
#                                               self._events_file
#                                               )

            spec.switch_write_focus (MLPRegions.EVENTS.value)

            # open the events file
            _vf = open (self._events_file, "rb")

            # read the data into a numpy array and put in spec
            _ev = np.fromfile (_vf, np.uint8)
            _vf.close ()
            for byte in _ev:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the routing region
        spec.reserve_memory_region (
            MLPRegions.ROUTING.value, self._N_KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys (fwd, bkp)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fwd_link), data_type = DataType.UINT32)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.bkp_link), data_type = DataType.UINT32)
        spec.write_value (0, data_type = DataType.UINT32)
        spec.write_value (0, data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
