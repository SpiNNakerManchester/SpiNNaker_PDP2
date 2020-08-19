import os
import struct

import spinnaker_graph_front_end as gfe

from pacman.model.graphs.machine import MachineEdge

from spinn_pdp2.input_vertex     import InputVertex
from spinn_pdp2.sum_vertex       import SumVertex
from spinn_pdp2.threshold_vertex import ThresholdVertex
from spinn_pdp2.weight_vertex    import WeightVertex
from spinn_pdp2.mlp_types        import MLPGroupTypes, MLPConstants, \
    MLPVarSizeRecordings, MLPConstSizeRecordings, MLPExtraRecordings
from spinn_pdp2.mlp_group        import MLPGroup
from spinn_pdp2.mlp_link         import MLPLink
from spinn_pdp2.mlp_examples     import MLPExampleSet


class MLPNetwork():
    """ top-level MLP network object.
            contains groups, links and
            and network-level parameters.
    """

    def __init__(self,
                net_type,
                intervals = 1,
                ticks_per_interval = 1,
                ):
        """
        """
        # assign network parameter values from arguments
        self._net_type           = net_type.value
        self._intervals          = intervals
        self._ticks_per_interval = ticks_per_interval

        # default network parameter values
        self._global_max_ticks = (intervals * ticks_per_interval) + 1
        self._train_group_crit = None
        self._test_group_crit  = None
        self._learning_rate    = MLPConstants.DEF_LEARNING_RATE
        self._weight_decay     = MLPConstants.DEF_WEIGHT_DECAY
        self._momentum         = MLPConstants.DEF_MOMENTUM
        self._update_function  = MLPConstants.DEF_UPDATE_FUNC
        self._num_updates      = MLPConstants.DEF_NUM_UPDATES
        self._num_examples     = None

        # default stage parameter values
        self._stg_update_function = MLPConstants.DEF_UPDATE_FUNC
        self._stg_epochs          = MLPConstants.DEF_NUM_UPDATES
        self._stg_examples        = None
        self._stg_reset           = True

        # default data recording options
        self._rec_test_results           = True
        self._rec_outputs                = True
        self._rec_example_last_tick_only = False

        # initialise lists of groups and links
        self.groups = []
        self.links  = []

        # initialise lists of INPUT and OUTPUT groups
        self.in_grps  = []
        self.out_grps = []

        # OUTPUT groups form a chain for convergence decision
        self._output_chain = []

        # track if initial weights have been loaded
        self._weights_rdy = False
        self._weights_loaded = False
        self._weights_file = None

        # initialise example set
        self._ex_set = None

        # create single-unit Bias group by default
        self._bias_group = self.group (units        = 1,
                                       group_type   = [MLPGroupTypes.BIAS],
                                       label        = "Bias"
                                       )

        # initialise machine graph parameters
        self._graph_rdy = False
        
        # keep track of the number of vertices in the graph
        self._num_vertices = 0

        # keep track of the number of partitions
        self.partitions = 0

        # keep track of the current execution stage
        self._stage_id = 0

        # keep track if errors have occurred
        self._aborted = False


    @property
    def net_type (self):
        return self._net_type

    @property
    def training (self):
        return self._training

    @property
    def num_epochs (self):
        return self._num_epochs

    @property
    def num_examples (self):
        return self._num_examples

    @property
    def ticks_per_int (self):
        return self._ticks_per_interval

    @property
    def global_max_ticks (self):
        return self._global_max_ticks

    @property
    def rec_test_results (self):
        return self._rec_test_results

    @property
    def rec_outputs (self):
        return self._rec_outputs

    @property
    def rec_example_last_tick_only (self):
        return self._rec_example_last_tick_only

    @property
    def num_write_blocks (self):
        return self._num_write_blks

    @property
    def output_chain (self):
        return self._output_chain

    @property
    def bias_group (self):
        return self._bias_group

    @property
    def network_config (self):
        """ returns a packed string that corresponds to
            (C struct) network_conf in mlp_types.h:

            typedef struct network_conf
            {
              uchar net_type;
              uint  ticks_per_int;
              uint  global_max_ticks;
              uint  num_write_blks;
            } network_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        return struct.pack("<B3x3I",
                           self._net_type,
                           self._ticks_per_interval,
                           self._global_max_ticks,
                           self._num_write_blks
                           )


    @property
    def stage_config (self):
        """ returns a packed string that corresponds to
            (C struct) stage_conf in mlp_types.h:

            typedef struct stage_conf
            {
              uchar stage_id;         // stage identifier
              uchar training;         // stage mode: train (1) or test (0)
              uchar update_function;  // weight update function in this stage
              uchar reset;            // reset example index at stage start?
              uint  num_examples;     // examples to run in this stage
              uint  num_epochs;       // training epochs in this stage
            } stage_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # set the update function to use in this stage
        if self._stg_update_function is not None:
            _update_function = self._stg_update_function
        else:
            _update_function = self._update_function

        # set the number of examples to use in this stage
        if self._stg_examples is not None:
            _num_examples = self._stg_examples
        else:
            _num_examples = self._ex_set.num_examples

        # set the number of epochs to run in this stage
        if self._stg_epochs is not None:
            _num_epochs = self._stg_epochs
        else:
            _num_epochs = self._num_updates

        return struct.pack("<4B2I",
                           self._stage_id,
                           self.training,
                           _update_function.value,
                           self._stg_reset,
                           _num_examples,
                           _num_epochs
                           )


    def group (self,
               units        = None,
               group_type   = [MLPGroupTypes.HIDDEN],
               input_funcs  = None,
               output_funcs = None,
               label        = None
               ):
        """ add a group to the network

        :param units: number of units that form the group
        :param group_type: list of Lens-style group types
        :param input_funcs: functions applied in the input pipeline
        :param output_funcs: functions applied in the output pipeline
        :param label: human-readable group identifier

        :type units: unsigned integer
        :type group_type: enum MLPGroupTypes
        :type input_funcs: enum MLPInputProcs
        :type output_funcs: enum MLPOutputProcs
        :type label: string

        :return: a new group object
        """
        # machine graph needs rebuilding
        self._graph_rdy = False
        
        _id = len (self.groups)

        # set properties for OUTPUT group
        if (MLPGroupTypes.OUTPUT in group_type):
            _write_blk = len (self.output_chain)
            if len (self.output_chain):
                _is_first_out = 0
            else:
                _is_first_out = 1
        else:
            _write_blk    = 0
            _is_first_out = 0

        # instantiate a new group
        _group = MLPGroup (_id,
                           units        = units,
                           gtype        = group_type,
                           input_funcs  = input_funcs,
                           output_funcs = output_funcs,
                           write_blk    = _write_blk,
                           is_first_out = _is_first_out,
                           label        = label
                           )

        # append new group to network list
        self.groups.append (_group)

        print (f"adding group {label} [total: {len(self.groups)}]")

        # if it's an INPUT group add to list
        if (MLPGroupTypes.INPUT in group_type):
            self.in_grps.append (_group)

        # if it's an OUTPUT group add to list and to the tail of the chain
        if (MLPGroupTypes.OUTPUT in group_type):
            self.out_grps.append (_group)
            self.output_chain.append (_group)

        # OUTPUT and HIDDEN groups instantiate BIAS links by default
        if (MLPGroupTypes.OUTPUT in group_type or\
            MLPGroupTypes.HIDDEN in group_type):
            self.link (self.bias_group, _group)

        # a new group forces reloading of initial weights file
        self._weights_loaded = False

        return _group


    def link (self,
              pre_link_group = None,
              post_link_group = None,
              label = None
              ):
        """ add a link to the network

        :param pre_link_group: link source group
        :param post_link_group: link destination group
        :param label: human-readable link identifier

        :type pre_link_group: MLPGroup
        :type post_link_group: MLPGroup
        :type label: string

        :return: a new link object
        """
        # machine graph needs rebuilding
        self._graph_rdy = False
        
        # check that enough data is provided
        if (pre_link_group is None) or (post_link_group is None):
            print ("error: pre- and post-link groups required")
            return None

        if label is None:
            _label = "{}-{}".format (pre_link_group.label,
                                     post_link_group.label
                                     )
        else:
            _label = label

        # instantiate a new link
        _link = MLPLink (pre_link_group  = pre_link_group,
                         post_link_group = post_link_group,
                         label           = _label
                         )

        # add new link to the network list
        self.links.append (_link)

        print (f"adding link from {pre_link_group.label} to "
               f"{post_link_group.label} [total: {len (self.links)}]"
               )

        # a new link forces reloading of initial weights file
        self._weights_loaded = False

        return _link


    def example_set (self,
                     label = None
                     ):
        """ add an example set to the network

        :param example_set: example set to be added

        :type example set: MLPExampleSet
        """
        self.label = label

        # instantiate a new example set
        _set = MLPExampleSet (label = label,
                              max_time = self._intervals
                             )

        # add example set to the network list
        self._ex_set = _set

        print (f"adding example set {label}")

        return _set


    def set (self,
             num_updates      = None,
             train_group_crit = None,
             test_group_crit  = None,
             learning_rate    = None,
             weight_decay     = None,
             momentum         = None
             ):
        """ set a network parameter to the given value

        :param num_updates: number of training epochs to be done
        :param train_group_crit: criterion used to stop training
        :param test_group_crit: criterion used to stop testing
        :param learning_rate: amount used to scale deltas when updating weights
        :param weight_decay: amount by which weights are scaled after being updated
        :param momentum: the carryover of previous weight changes to the new step

        :type num_updates: unsigned integer
        :type train_group_crit: float
        :type test_group_crit: float
        :type learning_rate: float
        :type weight_decay: float
        :type momentum: float
        """
        if num_updates is not None:
            print (f"setting num_updates to {num_updates}")
            self._num_updates = num_updates

        if train_group_crit is not None:
            print (f"setting train_group_crit to {train_group_crit}")
            self._train_group_crit = train_group_crit

        if test_group_crit is not None:
            print (f"setting test_group_crit to {test_group_crit}")
            self._test_group_crit = test_group_crit

        if learning_rate is not None:
            print (f"setting learning_rate to {learning_rate}")
            self._learning_rate = learning_rate

        if weight_decay is not None:
            print (f"setting weight_decay to {weight_decay}")
            self._weight_decay = weight_decay

        if momentum is not None:
            print (f"setting momentum to {momentum}")
            self._momentum = momentum


    def recording_options (self,
             rec_test_results           = None,
             rec_outputs                = None,
             rec_example_last_tick_only = None
             ):
        """ set data recording options

        :param rec_test_results: record test results
        :param rec_outputs: record unit outputs
        :param rec_example_last_tick_only: record unit outputs only for
                                            last tick of examples

        :type rec_test_results: boolean
        :type rec_outputs: boolean
        :type rec_example_last_tick_only: boolean
        """
        #TODO: changing recording options between stages not currently supported
        if self._stage_id:
            print ("\n--------------------------------------------------")
            print ("warning: new recording options ignored - cannot change between stages")
            print ("--------------------------------------------------\n")
            return

        if rec_test_results is not None:
            print (f"setting rec_test_results to {rec_test_results}")
            self._rec_test_results = rec_test_results
        else:
            print (f"rec_test_results pre-set to {self._rec_test_results}")

        if rec_outputs is not None:
            print (f"setting rec_outputs to {rec_outputs}")
            self._rec_outputs = rec_outputs
        else:
            print (f"rec_outputs pre-set to {self._rec_outputs}")

        if rec_example_last_tick_only is not None:
            print (f"setting rec_example_last_tick_only to {rec_example_last_tick_only}")
            self._rec_example_last_tick_only = rec_example_last_tick_only
        else:
            print (f"rec_example_last_tick_only pre-set to {self._rec_example_last_tick_only}")


    def read_Lens_weights_file (self,
                                weights_file
                                ):
        """ reads a Lens-style weights file

        Lens online manual @ CMU:
            https://ni.cmu.edu/~plaut/Lens/Manual/

        File format:

        <I magic-weight-cookie>
        <I total-number-of-links>
        <I num-values>
        <I totalUpdates>
        for each group:
          for each unit in group:
            for each incoming link to unit:
              <R link-weight>
        	  if num-values >= 2:
                <R link-lastWeightDelta>
        	  if num-values >= 3:
                <R link-lastValue>
        """
        # check if file exists
        if os.path.isfile (weights_file):
            self._weights_file = weights_file
        elif os.path.isfile ("data/{}".format (weights_file)):
            self._weights_file = "data/{}".format (weights_file)
        else:
            self._weights_file = None
            print (f"error: cannot open weights file: {weights_file}")
            return False

        print ("reading Lens-style weights file")

        # compute the number of expected weights in the file
        _num_wts = 0
        for to_grp in self.groups:
            for frm_grp in to_grp.links_from:
                _num_wts = _num_wts + to_grp.units * frm_grp.units

        # check that it is the correct file type
        _wf = open (self._weights_file, "r")

        if int (_wf.readline ()) != MLPConstants.LENS_WEIGHT_MAGIC_COOKIE:
            print ("error: incorrect weights file type")
            _wf.close ()
            return False

        # check that the file contains the right number of weights
        if int (_wf.readline ()) != _num_wts:
            print ("error: incorrect number of weights "
                   f"in file; expected {_num_wts}"
                   )
            _wf.close ()
            return False

        # read weights from file and store them in the corresponding group
        _num_values = int (_wf.readline ())
        _ = _wf.readline ()  # discard number of updates

        for grp in self.groups:
            # create an empty weight list for every possible link
            for fgrp in self.groups:
                grp.weights[fgrp] = []

            # populate weight lists from file
            # lists store weights in column-major order
            for _ in range (grp.units):
                # read weight if link exists (read in the correct order!)
                for fgrp in self.groups:
                    if fgrp in grp.links_from:
                        for _ in range (fgrp.units):
                            grp.weights[fgrp].append (float (_wf.readline()))
                            if _num_values >= 2:
                                _ = _wf.readline ()  # discard
                            if _num_values >= 3:
                                _ = _wf.readline ()  # discard

        # clean up
        _wf.close ()

        # mark weights file as loaded
        self._weights_loaded = True

        return True


    def write_Lens_output_file (self,
                                output_file
                                ):
        """ writes a Lens-style output file

            Lens online manual @ CMU:
                https://ni.cmu.edu/~plaut/Lens/Manual/

            File format:

            for each example:
              <I total-updates> <I example-number>
              <I ticks-on-example> <I num-groups>
              for each tick on the example:
                <I tick-number> <I event-number>
            for each WRITE_OUTPUTS group:
                  <I num-units> <B targets?>
              for each unit:
                    <R output-value> <R target-value>

            collects recorded tick data corresponding to
            (C struct) tick_record in mlp_types.h:

            typedef struct tick_record {
              uint epoch;    // current epoch
              uint example;  // current example
              uint event;    // current event
              uint tick;     // current tick
            } tick_record_t;

            collects recorded output data corresponding to
            (C type) short_activ_t in mlp_types.h:

            typedef short short_activ_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        if not self._rec_outputs:
            print ("\n--------------------------------------------------")
            print ("warning: file write aborted - outputs not recorded")
            print ("--------------------------------------------------\n")
            return

        if not self._aborted:
            with open(output_file, 'w') as f:
                # prepare to retrieve recorded data
                TICK_DATA_FORMAT = "<4I"
                TICK_DATA_SIZE = struct.calcsize(TICK_DATA_FORMAT)

                OUT_DATA_FORMATS = []
                OUT_DATA_SIZES = []       
                for g in self.output_chain:
                    OUT_DATA_FORMATS.append ("<{}H".format (g.units))
                    OUT_DATA_SIZES.append (struct.calcsize("<{}H".format (g.units)))

                # retrieve recorded tick_data from first output group
                g = self.out_grps[0]
                rec_tick_data = g.t_vertex.read (
                    gfe.placements().get_placement_of_vertex (g.t_vertex),
                    gfe.buffer_manager(), MLPExtraRecordings.TICK_DATA.value
                    )

                TOTAL_TICKS = len (rec_tick_data) // TICK_DATA_SIZE

                # retrieve recorded outputs from every output group
                rec_outputs = [None] * len (self.out_grps)
                for g in self.out_grps:
                    rec_outputs[g.write_blk] = g.t_vertex.read (
                        gfe.placements().get_placement_of_vertex (g.t_vertex),
                        gfe.buffer_manager(), MLPVarSizeRecordings.OUTPUTS.value
                        )

                # compute total ticks in first example
                #TODO: need to get actual value from simulation, not max value
                ticks_per_example = 0
                for ev in self._ex_set.examples[0].events:
                    # use event max_time if available or default to set max_time,
                    if (ev.max_time is None) or (self.max_time == float ('nan')):
                        max_time = int (self._ex_set.max_time)
                    else:
                        max_time = int (ev.max_time)

                    # compute number of ticks for max time,
                    ticks_per_example += (max_time + 1) * self._ticks_per_interval

                    # and limit to the global maximum if required
                    if ticks_per_example > self.global_max_ticks:
                        ticks_per_example = self.global_max_ticks

                # print recorded data in correct order
                current_epoch = -1
                for tk in range (TOTAL_TICKS):
                    (epoch, example, event, tick) = struct.unpack_from(
                        TICK_DATA_FORMAT,
                        rec_tick_data,
                        tk * TICK_DATA_SIZE
                        )

                    # check if starting new epoch
                    if (epoch != current_epoch):
                        current_epoch = epoch
                        current_example = -1

                    # check if starting new example
                    if (example != current_example):
                        # print first (implicit) tick data
                        f.write (f"{epoch} {example}\n")
                        f.write (f"{ticks_per_example} {len (self.out_grps)}\n")
                        f.write ("0 -1\n")
                        for g in self.output_chain:
                            f.write (f"{g.units} 1\n")
                            for _ in range (g.units):
                                f.write ("{:8.6f} {}\n".format (0, 0))

                        # compute event index
                        evt_inx = 0
                        for ex in range (example):
                            evt_inx += len (self._ex_set.examples[ex].events)

                        # and prepare for next 
                        current_example = example

                    # compute index into target array
                    tgt_inx = evt_inx + event

                    # print current tick data
                    f.write (f"{tick} {event}\n")

                    for g in self.output_chain:
                        # get group tick outputs
                        outputs = struct.unpack_from(
                            OUT_DATA_FORMATS[self.output_chain.index(g)],
                            rec_outputs[g.write_blk],
                            tk * OUT_DATA_SIZES[self.output_chain.index(g)]
                            )

                        # print outputs
                        if len (rec_outputs[g.write_blk]):
                            f.write (f"{g.units} 1\n")
                            tinx = tgt_inx * g.units
                            for u in range (g.units):
                                # outputs are s16.15 fixed-point numbers
                                out = (1.0 * outputs[u]) / (1.0 * (1 << 15))
                                t = g.targets[tinx + u]
                                if (t is None) or (t == float ('nan')):
                                    tgt = "-"
                                else:
                                    tgt = int(t)
                                f.write ("{:8.6f} {}\n".format (out, tgt))

            # prepare buffers for next stage
            gfe.buffer_manager().reset()


    def show_test_results (self):
        """ show stage test results corresponding to
            (C struct) test_results in mlp_types.h:

            typedef struct test_results {
            uint epochs_trained;
            uint examples_tested;
            uint ticks_tested;
            uint examples_correct;
            } test_results_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        if not self._rec_test_results:
            print ("\n--------------------------------------------------")
            print ("warning: test results not recorded")
            print ("--------------------------------------------------\n")
            return

        if not self._aborted:
            # prepare to retrieve recorded test results data
            TEST_RESULTS_FORMAT = "<4I"
            TEST_RESULTS_SIZE = struct.calcsize(TEST_RESULTS_FORMAT)

            # retrieve recorded tick_data from last output group
            g = self.out_grps[-1]
            rec_test_results = g.t_vertex.read (
                gfe.placements().get_placement_of_vertex (g.t_vertex),
                gfe.buffer_manager(), MLPConstSizeRecordings.TEST_RESULTS.value
                )

            if len (rec_test_results) >= TEST_RESULTS_SIZE:
                (epochs_trained, examples_tested, ticks_tested, examples_correct) = \
                    struct.unpack_from(TEST_RESULTS_FORMAT, rec_test_results, 0)
    
                print ("\n--------------------------------------------------")
                print ("stage {} Test results: {}, {}, {}, {}".format(
                    self._stage_id, epochs_trained, examples_tested,
                    ticks_tested, examples_correct
                    ))
                print ("--------------------------------------------------\n")


    def generate_machine_graph (self):
        """ generates a machine graph for the application graph
        """
        print ("generating machine graph")

        # path to binary files
        binaries_path = os.path.join(os.path.dirname(__file__), "..", "binaries")

        # setup the machine graph
        gfe.setup (model_binary_folder = binaries_path)

        # set the number of write blocks before generating vertices
        self._num_write_blks = len (self.output_chain)

        # compute number of partitions
        for grp in self.groups:
            self.partitions = self.partitions + grp.partitions

        # create associated weight, sum, input and threshold
        # machine vertices for every network group
        for grp in self.groups:
            # create one weight core per partition
            # of every (from_group, group) pair
            # NOTE: all-zero cores can be optimised out
            for from_grp in self.groups:
                for _tp in range (grp.partitions):
                    for _fp in range (from_grp.partitions):
                        wv = WeightVertex (self, grp, from_grp, _tp, _fp)
                        grp.w_vertices.append (wv)
                        gfe.add_machine_vertex_instance (wv)
                        self._num_vertices += 1

            # create one sum core per group
            sv = SumVertex (self, grp)
            grp.s_vertex = sv
            gfe.add_machine_vertex_instance (sv)
            self._num_vertices += 1

            # create one input core per group
            iv = InputVertex (self, grp)
            grp.i_vertex = iv
            gfe.add_machine_vertex_instance (iv)
            self._num_vertices += 1

            # create one threshold core per group
            tv = ThresholdVertex (self, grp)
            grp.t_vertex = tv
            gfe.add_machine_vertex_instance (tv)
            self._num_vertices += 1

        # create associated forward, backprop, link delta summation,
        # synchronisation and stop machine edges for every network group
        first = self.groups[0]
        for grp in self.groups:
            for w in grp.w_vertices:
                _frmg = w.from_group

                # create forward w to s links
                gfe.add_machine_edge_instance (MachineEdge (w, grp.s_vertex),
                                             w.fwd_link)

                # create forward t to w (multicast) links
                gfe.add_machine_edge_instance (MachineEdge (_frmg.t_vertex, w),
                                             _frmg.t_vertex.fwd_link[w.row_blk])

                # create backprop w to s links
                gfe.add_machine_edge_instance (MachineEdge (w, _frmg.s_vertex),
                                             w.bkp_link)

                # create backprop i to w (multicast) links
                gfe.add_machine_edge_instance (MachineEdge (grp.i_vertex, w),
                                             grp.i_vertex.bkp_link[w.col_blk])

                # create link delta summation w to s links
                gfe.add_machine_edge_instance (MachineEdge (w, grp.s_vertex),
                                             w.lds_link)

                # create link delta summation result s (first) to w links
                gfe.add_machine_edge_instance (MachineEdge (first.s_vertex, w),
                                             first.s_vertex.lds_link)

                # create example synchronisation s to w (multicast) links
                gfe.add_machine_edge_instance (MachineEdge (grp.s_vertex, w),
                                               grp.s_vertex.fds_link)

                if grp != _frmg:
                    gfe.add_machine_edge_instance (MachineEdge (_frmg.s_vertex, w),
                                                 _frmg.s_vertex.fds_link)

            # create forward s to i link
            gfe.add_machine_edge_instance (MachineEdge (grp.s_vertex,
                                                      grp.i_vertex),
                                         grp.s_vertex.fwd_link)

            # create backprop s to t link
            gfe.add_machine_edge_instance (MachineEdge (grp.s_vertex,
                                                      grp.t_vertex),
                                         grp.s_vertex.bkp_link)

            # create forward i to t link
            gfe.add_machine_edge_instance (MachineEdge (grp.i_vertex,
                                                      grp.t_vertex),
                                         grp.i_vertex.fwd_link)

            # create backprop t to i link
            gfe.add_machine_edge_instance (MachineEdge (grp.t_vertex,
                                                      grp.i_vertex),
                                         grp.t_vertex.bkp_link)

            # create link delta summation s to s links - all s cores
            # (except the first) send to the first s core
            if grp != first:
                print (f"Creating lds s-s edge from group {grp.label} "
                       f"to group {first.label}")
                gfe.add_machine_edge_instance (MachineEdge (grp.s_vertex,
                                                          first.s_vertex),
                                             grp.s_vertex.lds_link)

            # create stop links, if OUTPUT group
            if grp in self.output_chain:
                # if last OUTPUT group broadcast stop decision
                if grp == self.output_chain[-1]:
                    for stpg in self.groups:
                        # create stop links to all w cores
                        for w in stpg.w_vertices:
                            gfe.add_machine_edge_instance\
                              (MachineEdge (grp.t_vertex, w),
                               grp.t_vertex.stp_link)

                        # create stop links to all s cores
                        gfe.add_machine_edge_instance\
                         (MachineEdge (grp.t_vertex, stpg.s_vertex),\
                          grp.t_vertex.stp_link)

                        # create stop links to all i cores
                        gfe.add_machine_edge_instance\
                         (MachineEdge (grp.t_vertex, stpg.i_vertex),\
                          grp.t_vertex.stp_link)

                        # create stop links to t cores (no link to itself!)
                        if stpg != grp:
                            gfe.add_machine_edge_instance\
                             (MachineEdge (grp.t_vertex, stpg.t_vertex),\
                              grp.t_vertex.stp_link)
                else:
                    # create stop link to next OUTPUT group in chain
                    _inx  = self.output_chain.index (grp)
                    _stpg = self.output_chain[_inx + 1]
                    gfe.add_machine_edge_instance (MachineEdge (grp.t_vertex,
                                                              _stpg.t_vertex),
                                                 grp.t_vertex.stp_link)

        self._graph_rdy = True


    def train (self,
               update_function = None,
               num_updates = None
              ):
        """ do one stage in train mode
        """
        # set the update function to use in this stage
        #NOTE: sorted at configuration time - if not provided
        self._stg_update_function = update_function

        # set the number of epochs to run in this stage
        #NOTE: sorted at configuration time - if not provided
        self._stg_epochs = num_updates

        # sort the number of examples at configuration time
        self._stg_examples = None

        # always reset the example index at the start of training stage 
        self._stg_reset = True

        self._training = 1
        self.stage_run ()


    def test (self,
               num_examples = None,
               reset_examples = True
              ):
        """ do one stage in test mode
        """
        # sort the update function at configuration time
        self._stg_update_function = None

        # set the number of epochs to run in this stage
        self._stg_epochs = 1

        # set the number of examples to run in this stage
        #NOTE: sorted at configuration time - if not provided
        self._stg_examples = num_examples

        # reset the example index if requested
        self._stg_reset = reset_examples

        self._training = 0
        self.stage_run ()


    def stage_run (self):
        """ run a stage on application graph
        """
        self._aborted = False

        # check that no group is too big
        for grp in self.groups:
            if grp.units > MLPConstants.MAX_GRP_UNITS:
                print (f"run aborted: group {grp.label} has more than "
                       f"{MLPConstants.MAX_GRP_UNITS} units.")
                self._aborted = True
                return

        # cannot run unless weights file exists
        if self._weights_file is None:
            print ("run aborted: weights file not given")
            self._aborted = True
            return

        # may need to reload initial weights file if
        # application graph was modified after load
        if not self._weights_loaded:
            if not self.read_Lens_weights_file (self._weights_file):
                print ("run aborted: error reading weights file")
                self._aborted = True
                return

        # cannot run unless example set exists
        if self._ex_set is None:
            print ("run aborted: no example set")
            self._aborted = True
            return

        # cannot run unless examples have been loaded
        if not self._ex_set.examples_loaded:
            print ("run aborted: examples not loaded")
            self._aborted = True
            return

        # generate summary set, example and event data
        if not self._ex_set.examples_compiled:
            if self._ex_set.compile (self) == 0:
                print ("run aborted: error compiling example set")
                self._aborted = True
                return

        # generate machine graph - if needed
        if not self._graph_rdy:
            self.generate_machine_graph ()

        # run stage
        gfe.run_until_complete (self._stage_id)

        # show TEST RESULTS if available
        if self.rec_test_results and not self.training:
            self.show_test_results ()

        # prepare for next stage
        self._stage_id += 1


    def pause (self):
        """ pause execution to allow debugging
        """
        # pause until a key is pressed
        input ("network paused: press enter to continue")


    def end (self):
        """ clean up before exiting
        """
        if not self._aborted:
            print ("exit: application finished")
            # let the gfe clean up
            gfe.stop()
