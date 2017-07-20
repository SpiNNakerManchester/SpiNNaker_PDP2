import struct
import numpy as np
import os

from data_specification.enums.data_type import DataType

from pacman.executor.injection_decorator import inject_items

from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.decorators.overrides import overrides
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource
from pacman.model.resources.iptag_resource import IPtagResource

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

from mlp_network import MLPRegions, MLPOutputProcs


class ThresholdVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP threshold core
    """

    def __init__(self,
                 network=None,
                 group=None,
                 output_grp = 0,
                 input_grp = 0,
                 num_outputs = None,
                 fwd_sync_expect = None,
                 bkp_sync_expect = 0,
                 write_out = 0,
                 write_blk = 0,
                 out_integr_en = 0,
                 out_integr_dt = 0,
                 num_out_procs = 0,
                 procs_list = [MLPOutputProcs.OUT_NONE.value,\
                               MLPOutputProcs.OUT_NONE.value,\
                               MLPOutputProcs.OUT_NONE.value,\
                               MLPOutputProcs.OUT_NONE.value,\
                               MLPOutputProcs.OUT_NONE.value],
                 weak_clamp_strength = 0x00008000,
                 initOutput = 0x4000,
                 group_criterion = 0,
                 criterion_function = 0,
                 is_first_output_group = 0,
                 is_last_output_group = 0,
                 error_function = 0
                 ):
        """
        """

        # MLP network
        self._network = network
        self._group   = group

        MachineVertex.__init__(self, label =\
                               "t{} core".format (self._group))

        # threshold core-specific parameters
        self._output_grp            = output_grp
        self._input_grp             = input_grp
        self._num_outputs           = num_outputs
        self._fwd_sync_expect       = fwd_sync_expect
        self._bkp_sync_expect       = bkp_sync_expect
        self._write_out             = write_out
        self._write_blk             = write_blk
        self._out_integr_en         = out_integr_en
        self._out_integr_dt         = out_integr_dt
        self._num_out_procs         = num_out_procs
        self._procs_list            = procs_list
        self._weak_clamp_strength   = weak_clamp_strength
        self._initOutput            = initOutput
        self._group_criterion       = group_criterion
        self._criterion_function    = criterion_function
        self._is_first_output_group = is_first_output_group
        self._is_last_output_group  = is_last_output_group
        self._error_function        = error_function


        # forward and backprop link partition names
        self._fwd_link = "fwd_s{}".format (self._group)
        self._bkp_link = "bkp_s{}".format (self._group)
        self._stp_link = "stp_s{}".format (self._group)

        self._n_keys = 65536

        # binary, configuration and data files
        self._aplxFile = "binaries/threshold.aplx"
        self._inputsFile = "data/inputs_{}.dat".format (self._group)
        self._targetsFile = "data/targets_{}.dat".format (self._group)
        self._exSetFile = "data/example_set.dat"
        self._examplesFile = "data/examples.dat"
        self._eventsFile = "data/events.dat"

        self._N_NETWORK_CONFIGURATION_BYTES = \
            len ((self._network).config)

        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        self._N_INPUTS_CONFIGURATION_BYTES = \
            os.path.getsize (self._inputsFile) \
            if os.path.isfile (self._inputsFile) \
            else 0

        self._N_TARGETS_CONFIGURATION_BYTES = \
            os.path.getsize (self._targetsFile) \
            if os.path.isfile (self._targetsFile) \
            else 0

        self._N_EXAMPLE_SET_BYTES = \
            os.path.getsize (self._exSetFile) \
            if os.path.isfile (self._exSetFile) \
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
            self._N_TARGETS_CONFIGURATION_BYTES + \
            self._N_EXAMPLE_SET_BYTES + \
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
              uint          num_outputs;
              scoreboard_t  fwd_sync_expect;
              scoreboard_t  bkp_sync_expect;
              uchar         write_out;
              uint          write_blk;
              uchar         out_integr_en;
              fpreal        out_integr_dt;
              uint          num_out_procs;
              uint          procs_list[SPINN_NUM_OUT_PROCS];
              fpreal        weak_clamp_strength;
              short_activ_t initOutput;
              error_t       group_criterion;
              uchar         criterion_function;
              uchar         is_first_output_group;
              uchar         is_last_output_group;
              uchar         error_function;
            } t_conf_t;

            pack: standard sizes, little-endian byte-order,
            explicit padding
        """
        return struct.pack("<2B2x3IB3xIB3xi6Iih2xi4B",
                           self._output_grp,
                           self._input_grp,
                           self._num_outputs,
                           self._fwd_sync_expect,
                           self._bkp_sync_expect,
                           self._write_out,
                           self._write_blk,
                           self._out_integr_en,
                           self._out_integr_dt,
                           self._num_out_procs,
                           self._procs_list [0],
                           self._procs_list [1],
                           self._procs_list [2],
                           self._procs_list [3],
                           self._procs_list [4],
                           self._weak_clamp_strength,
                           self._initOutput,
                           self._group_criterion,
                           self._criterion_function,
                           self._is_first_output_group,
                           self._is_last_output_group,
                           self._error_function
                           )
    @property
    @overrides (MachineVertex.resources_required)
    def resources_required (self):

        resources = ResourceContainer (
            sdram = SDRAMResource (self._sdram_usage),
            iptags = [IPtagResource (ip_address = "localhost",
                                    port = 17896,
                                    strip_sdp = False)]
            )
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

        # Reserve and write the target data region
        if os.path.isfile (self._targetsFile):
            spec.reserve_memory_region (
                MLPRegions.TARGETS.value,
                self._N_TARGETS_CONFIGURATION_BYTES)

            spec.switch_write_focus (MLPRegions.TARGETS.value)

            # open target data file
            targets_file = open (self._targetsFile, "rb")

            # read the data into a numpy array and put in spec
            tc = np.fromfile (targets_file, np.uint8)
            for byte in tc:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the example set region
        if os.path.isfile (self._exSetFile):
            spec.reserve_memory_region (
                MLPRegions.EXAMPLE_SET.value,
                self._N_EXAMPLE_SET_BYTES)

            spec.switch_write_focus (MLPRegions.EXAMPLE_SET.value)

            # open the example set file
            ex_set_file = open (self._exSetFile, "rb")

            # read the data into a numpy array and put in spec
            es = np.fromfile (ex_set_file, np.uint8)
            for byte in es:
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

        # write link keys (fwd, bkp, fds, stp)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._fwd_link), data_type = DataType.UINT32)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._bkp_link), data_type = DataType.UINT32)
        spec.write_value (0, data_type = DataType.UINT32)
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self._stp_link), data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
