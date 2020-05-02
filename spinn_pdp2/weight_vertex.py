import struct

from data_specification.enums.data_type import DataType

from pacman.executor.injection_decorator import inject_items

from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.decorators.overrides import overrides
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource

from spinn_front_end_common.utilities.utility_objs \
    import ExecutableType
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models\
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models\
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

from mlp_types import MLPRegions, MLPConstants


class WeightVertex(
        MachineVertex,
        AbstractHasAssociatedBinary,
        AbstractProvidesNKeysForPartition,
        AbstractGeneratesDataSpecification):
    """ A vertex to implement an MLP input core
    """

    def __init__(self,
                 network,
                 group,
                 from_group,
                 col_blk,
                 row_blk
                 ):
        """
        """
        MachineVertex.__init__(self, label =\
                               "w{}_{}_{}_{} core".format (
                                   group.id, from_group.id,
                                   row_blk, col_blk)
                               )

        # application-level data
        self._network    = network
        self._group      = group
        self._from_group = from_group
        self._col_blk    = col_blk
        self._row_blk    = row_blk
        self._ex_cfg     = network._ex_set.example_config

        # compute number of rows and columns
        if self._row_blk != (self.from_group.partitions - 1):
            self._num_rows = MLPConstants.MAX_BLK_UNITS
        else:
            _r = self.from_group.units % MLPConstants.MAX_BLK_UNITS
            if _r == 0:
                self._num_rows = MLPConstants.MAX_BLK_UNITS
            else:
                self._num_rows = _r

        if self._col_blk != (self.group.partitions - 1):
            self._num_cols = MLPConstants.MAX_BLK_UNITS
        else:
            _r = self.group.units % MLPConstants.MAX_BLK_UNITS
            if _r == 0:
                self._num_cols = MLPConstants.MAX_BLK_UNITS
            else:
                self._num_cols = _r

        # forward, backprop, synchronisation, and link delta summation link partition names
        self._fwd_link = "fwd_w{}_{}".format (self.group.id,
                                              self.from_group.id)
        self._bkp_link = "bkp_w{}_{}".format (self.group.id,
                                              self.from_group.id)
        self._fds_link = "fds_w{}_{}".format (self.group.id,
                                              self.from_group.id)
        self._lds_link = "lds_w{}_{}".format (self.group.id,
                                              self.from_group.id)

        # reserve key space for every link
        self._n_keys = MLPConstants.KEY_SPACE_SIZE

        # choose weight core-specific parameters
        if len (self.group.weights[self.from_group]):
            if self.group.learning_rate is not None:
                self.learning_rate = self.group.learning_rate
            elif network._learning_rate is not None:
                self.learning_rate = network._learning_rate
            else:
                self.learning_rate = MLPConstants.DEF_LEARNING_RATE

            if self.group.weight_decay is not None:
                self.weight_decay = self.group.weight_decay
            elif network._weight_decay is not None:
                self.weight_decay = network._weight_decay
            else:
                self.weight_decay = MLPConstants.DEF_WEIGHT_DECAY

            if self.group.momentum is not None:
                self.momentum = self.group.momentum
            elif network._momentum is not None:
                self.momentum = network._momentum
            else:
                self.momentum = MLPConstants.DEF_MOMENTUM
        else:
            self.learning_rate = 0
            self.weight_decay = 0
            self.momentum = 0

        # weight update function
        self.update_function = network._update_function

        # binary, configuration and data files
        self._aplx_file = "binaries/weight.aplx"

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

        # each weight is an integer
        self._N_WEIGHTS_BYTES = \
            self.group.units * self.from_group.units * int_size

        # keys are integers
        self._N_KEYS_BYTES = MLPConstants.NUM_KEYS_REQ * int_size

        self._sdram_usage = (
            self._N_NETWORK_CONFIGURATION_BYTES + \
            self._N_CORE_CONFIGURATION_BYTES + \
            self._N_EXAMPLES_BYTES + \
            self._N_WEIGHTS_BYTES + \
            self._N_KEYS_BYTES
        )

    @property
    def group (self):
        return self._group

    @property
    def from_group (self):
        return self._from_group

    @property
    def fwd_link (self):
        return self._fwd_link

    @property
    def bkp_link (self):
        return self._bkp_link

    @property
    def fds_link (self):
        return self._fds_link

    @property
    def lds_link (self):
        return self._lds_link

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
            print "warning: input weight >= {}".format (MLPConstants.WF_MAX)
        elif wt_float <= MLPConstants.WF_MIN:
            wtemp = MLPConstants.WF_MIN;
            print "warning: input weight <= {}".format (MLPConstants.WF_MIN)
        else:
            wtemp = wt_float

        # return an MLP fixed-point weight_t
        return (int (wtemp * (1 << MLPConstants.WEIGHT_SHIFT)))

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) w_conf in mlp_types.h:

            typedef struct w_conf
            {
              uint           num_rows;
              uint           num_cols;
              uint           row_blk;
              uint           col_blk;
              short_fpreal_t learningRate;
              short_fpreal_t weightDecay;
              short_fpreal_t momentum;
              uchar          update_function;
            } w_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # learning_rate is an MLP short fixed-point fpreal
        learning_rate = int (self.learning_rate *\
                              (1 << MLPConstants.SHORT_FPREAL_SHIFT))

        # weight_decay is an MLP short fixed-point fpreal
        weight_decay = int (self.weight_decay *\
                              (1 << MLPConstants.SHORT_FPREAL_SHIFT))

        # momentum is an MLP short fixed-point fpreal
        momentum = int (self.momentum *\
                              (1 << MLPConstants.SHORT_FPREAL_SHIFT))

        return struct.pack ("<4I3hBx",
                            self._num_rows,
                            self._num_cols,
                            self._row_blk,
                            self._col_blk,
                            learning_rate & 0xffff,
                            weight_decay & 0xffff,
                            momentum & 0xffff,
                            self.update_function.value & 0xff
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
        return ExecutableType.SYNC

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

        # Reserve and write the weights region
        spec.reserve_memory_region (MLPRegions.WEIGHTS.value,
                                    self._N_WEIGHTS_BYTES)

        spec.switch_write_focus (MLPRegions.WEIGHTS.value)

        # weight matrix is kept in column-major order
        # and has to be written out in row-major order
        _wts = self.group.weights[self.from_group]
        _nrows = self.from_group.units
        _nr = self._num_rows
        _nc = self._num_cols
        _rb = self._row_blk * MLPConstants.MAX_BLK_UNITS
        _cb = self._col_blk * MLPConstants.MAX_BLK_UNITS
        if len (_wts):
            for _r in range (_nr):
                for _c in range (_nc):
                    _wt = self.cast_float_to_weight (
                        _wts[(_cb + _c) * _nrows + (_rb + _r)])
                    spec.write_value (_wt, data_type = DataType.INT32)
        else:
            for _ in range (_nr * _nc):
                spec.write_value (0, data_type = DataType.INT32)

        # Reserve and write the routing region
        spec.reserve_memory_region (MLPRegions.ROUTING.value,
                                    self._N_KEYS_BYTES)

        spec.switch_write_focus (MLPRegions.ROUTING.value)

        # write link keys: fwd, bkp, fds, stop (padding), and lds
        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fwd_link), data_type = DataType.UINT32)

        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.bkp_link), data_type = DataType.UINT32)

        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.fds_link), data_type = DataType.UINT32)

        spec.write_value (0, data_type = DataType.UINT32)

        spec.write_value (routing_info.get_first_key_from_pre_vertex (
            self, self.lds_link), data_type = DataType.UINT32)

        # End the specification
        spec.end_specification ()
