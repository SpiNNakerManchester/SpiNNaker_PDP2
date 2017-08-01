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

from mlp_types import MLPRegions, MLPConstants


class ThresholdVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP threshold core
    """

    def __init__(self,
                 network,
                 group,
                 fwd_sync_expect = None,
                 bkp_sync_expect = 0,
                 is_last_out     = None
                 ):
        """
        """
        MachineVertex.__init__(self, label =\
                               "t{} core".format (group.id))

        # application-level data
        self._network               = network
        self._group                 = group

        # forward, backprop and stop link partition names
        self._fwd_link = "fwd_s{}".format (self.group.id)
        self._bkp_link = "bkp_s{}".format (self.group.id)
        self._stp_link = "stp_s{}".format (self.group.id)

        # threshold core-specific parameters
        self._fwd_sync_expect       = fwd_sync_expect
        self._bkp_sync_expect       = bkp_sync_expect
        self._is_last_output_group  = is_last_out
        self._out_integr_dt = int ((1.0 / network.ticks_per_int) *\
                                   (1 << 16))

        # reserve a 16-bit key space in every link
        self._n_keys = MLPConstants.KEY_SPACE_SIZE

        # binary, configuration and data files
        self._aplx_file     = "binaries/threshold.aplx"
        self._inputs_file   = "data/inputs_{}.dat".\
                                format (self.group.id + 2) #lap
        self._targets_file  = "data/targets_{}.dat".\
                                format (self.group.id + 2) #lap
        self._exSet_file    = "data/example_set.dat"
        self._examples_file = "data/examples.dat"
        self._events_file   = "data/events.dat"

        # find out the size of an integer!
        _data_int=DataType.INT32
        int_size = _data_int.size

        self._N_NETWORK_CONFIGURATION_BYTES = \
            len (self._network.config)

        self._N_CORE_CONFIGURATION_BYTES = \
            len (self.config)

        self._N_INPUTS_BYTES = \
            os.path.getsize (self._inputs_file) \
            if os.path.isfile (self._inputs_file) \
            else 0

        self._N_TARGETS_CONFIGURATION_BYTES = \
            os.path.getsize (self._targets_file) \
            if os.path.isfile (self._targets_file) \
            else 0

        self._N_EXAMPLE_SET_BYTES = \
            os.path.getsize (self._exSet_file) \
            if os.path.isfile (self._exSet_file) \
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
            self._N_TARGETS_CONFIGURATION_BYTES + \
            self._N_EXAMPLE_SET_BYTES + \
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
              uint          num_units;
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
        return struct.pack ("<2B2x3IB3xIB3xi6Iih2xi4B",
                            self.group.output_grp & 0xff,
                            self.group.input_grp & 0xff,
                            self.group.units,
                            self._fwd_sync_expect,
                            self._bkp_sync_expect,
                            self.group.write_out & 0xff,
                            self.group.write_blk,
                            self.group.out_integr_en & 0xff,
                            self._out_integr_dt,
                            self.group.num_out_procs,
                            self.group.out_procs_list[0].value,
                            self.group.out_procs_list[1].value,
                            self.group.out_procs_list[2].value,
                            self.group.out_procs_list[3].value,
                            self.group.out_procs_list[4].value,
                            self.group.weak_clamp_strength,
                            self.group.init_output,
                            self.group.group_criterion,
                            self.group.criterion_function.value & 0xff,
                            self.group.is_first_out & 0xff,
                            self._is_last_output_group & 0xff,
                            self.group.error_function.value & 0xff
                            )
    @property
    @overrides (MachineVertex.resources_required)
    def resources_required (self):

        resources = ResourceContainer (
            sdram  = SDRAMResource (self._sdram_usage),
            iptags = [IPtagResource (ip_address = "localhost",
                                    port        = 17896,
                                    strip_sdp   = False)]
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

#             print "tv-{}: reading {}".format (self.group.id,
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

        # Reserve and write the target data region
        if os.path.isfile (self._targets_file):
            spec.reserve_memory_region (
                MLPRegions.TARGETS.value,
                self._N_TARGETS_CONFIGURATION_BYTES)

#             print "tv-{}: reading {}".format (self.group.id,
#                                               self._targets_file
#                                               )

            spec.switch_write_focus (MLPRegions.TARGETS.value)

            # open target data file
            _tf = open (self._targets_file, "rb")

            # read the data into a numpy array and put in spec
            _tc = np.fromfile (_tf, np.uint8)
            _tf.close ()
            for byte in _tc:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the example set region
        if os.path.isfile (self._exSet_file):
            spec.reserve_memory_region (
                MLPRegions.EXAMPLE_SET.value,
                self._N_EXAMPLE_SET_BYTES)

#             print "tv-{}: reading {}".format (self.group.id,
#                                               self._exSet_file
#                                               )

            spec.switch_write_focus (MLPRegions.EXAMPLE_SET.value)

            # open the example set file
            _sf = open (self._exSet_file, "rb")

            # read the data into a numpy array and put in spec
            _es = np.fromfile (_sf, np.uint8)
            _sf.close ()
            for byte in _es:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the examples region
        if os.path.isfile (self._examples_file):
            spec.reserve_memory_region (
                MLPRegions.EXAMPLES.value,
                self._N_EXAMPLES_BYTES)

#             print "tv-{}: reading {}".format (self.group.id,
#                                               self._examples_file
#                                               )

            spec.switch_write_focus (MLPRegions.EXAMPLES.value)

            # open the examples file
            _xf = open (self._examples_file, "rb")

            # read the data into a numpy array and put in spec
            _ex = np.fromfile (_xf, np.uint8)
            _xf.close ()
            for byte in _ex:
                spec.write_value (byte, data_type=DataType.UINT8)

        # Reserve and write the events region
        if os.path.isfile (self._events_file):
            spec.reserve_memory_region (
                MLPRegions.EVENTS.value,
                self._N_EVENTS_BYTES)

#             print "tv-{}: reading {}".format (self.group.id,
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

        # write link keys (fwd, bkp, fds, stp)
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


        # End the specification
        spec.end_specification ()
