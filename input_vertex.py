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

from mlp_network import MLPRegions, MLPInputProcs


class InputVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP input core
    """

    def __init__(self,
                 network=None,
                 group=None,
                 output_grp = 0,
                 input_grp = 0,
                 num_nets = None,
                 num_in_procs = 0,
                 procs_list = [MLPInputProcs.IN_NONE.value,\
                               MLPInputProcs.IN_NONE.value],
                 in_integr_en = 0,
                 in_integr_dt = 0,
                 soft_clamp_strength = 0x00008000,
                 initNets = 0,
                 initOutput = 0x4000
                 ):
        """
        """

        # MLP network
        self._network = network
        self._group   = group

        MachineVertex.__init__(self, label =\
                               "i{} core".format (self._group))

        # input core-specific parameters
        self._output_grp          = output_grp
        self._input_grp           = input_grp
        self._num_nets            = num_nets
        self._num_in_procs        = num_in_procs
        self._procs_list          = procs_list
        self._in_integr_en        = in_integr_en
        self._in_integr_dt        = in_integr_dt
        self._soft_clamp_strength = soft_clamp_strength
        self._initNets            = initNets
        self._initOutput          = initOutput

        # forward and backprop link partition names
        self._fwd_link = "fwd_i{}".format (self._group)
        self._bkp_link = "bkp_i{}".format (self._group)

        self._n_keys = 65536

        # binary, configuration and data files
        self._aplxFile = "binaries/input.aplx"
        self._inputsFile = "data/inputs_{}.dat".format (self._group)
        self._examplesFile = "data/examples.dat"
        self._eventsFile = "data/events.dat"

        # size in bytes of the data in the regions
        self._N_NETWORK_CONFIGURATION_BYTES = \
            len ((self._network).config)

        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        self._N_INPUTS_CONFIGURATION_BYTES = \
            os.path.getsize (self._inputsFile) \
            if os.path.isfile (self._inputsFile) \
            else 0

        self._N_EXAMPLES_BYTES = \
            os.path.getsize (self._examplesFile) \
            if os.path.isfile (self._examplesFile) \
            else 0

        self._N_EVENTS_BYTES = \
            os.path.getsize (self._eventsFile) \
            if os.path.isfile (self._eventsFile) \
            else 0

        self._N_KEY_BYTES = 16

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_INPUTS_CONFIGURATION_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_EVENTS_BYTES + \
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
            (C struct) i_conf in mlp_types.h:

            typedef struct i_conf
            {
              uchar         output_grp;
              uchar         input_grp;
              uint          num_nets;
              uint          num_in_procs;
              uint          procs_list[SPINN_NUM_IN_PROCS];
              uchar         in_integr_en;
              fpreal        in_integr_dt;
              fpreal        soft_clamp_strength;
              net_t         initNets;
              short_activ_t initOutput;
            } i_conf_t;

            pack: standard sizes, little-endian byte-order,
            explicit padding
        """
        return struct.pack("<2B2x4IB3x3ih2x",
                           self._output_grp,
                           self._input_grp,
                           self._num_nets,
                           self._num_in_procs,
                           self._procs_list[0],
                           self._procs_list[1],
                           self._in_integr_en,
                           self._in_integr_dt,
                           self._soft_clamp_strength,
                           self._initNets,
                           self._initOutput & 0xffff
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
        spec.reserve_memory_region (
            MLPRegions.CORE.value, self._N_CORE_CONFIGURATION_BYTES)

        spec.switch_write_focus (MLPRegions.CORE.value)

        # write the core configuration into spec
        for c in self.config:
            spec.write_value (ord (c), data_type=DataType.UINT8)


        # Reserve and write the input data region
        if os.path.isfile (self._inputsFile):
            spec.reserve_memory_region (
                MLPRegions.INPUTS.value,
                self._N_INPUTS_CONFIGURATION_BYTES)

            spec.switch_write_focus (MLPRegions.INPUTS.value)

            # open input data file
            inputs_file = open (self._inputsFile, "rb")

            # read the data into a numpy array and put in spec
            ic = np.fromfile (inputs_file, np.uint8)
            for byte in ic:
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

        # Reserve and write the events region
        if os.path.isfile (self._eventsFile):
            spec.reserve_memory_region (
                MLPRegions.EVENTS.value,
                self._N_EVENTS_BYTES)

            spec.switch_write_focus (MLPRegions.EVENTS.value)

            # open the events file
            ev_file = open (self._eventsFile, "rb")

            # read the data into a numpy array and put in spec
            ev = np.fromfile (ev_file, np.uint8)
            for byte in ev:
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
