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

        # keep track of the number of subgroups
        self.subgroups = 0

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
    def ex_set (self):
        return self._ex_set

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
    def train_group_crit (self):
        return self._train_group_crit

    @property
    def test_group_crit (self):
        return self._test_group_crit

    @property
    def learning_rate (self):
        return self._learning_rate

    @property
    def weight_decay (self):
        return self._weight_decay

    @property
    def momentum (self):
        return self._momentum

    @property
    def update_function (self):
        return self._update_function

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
            } network_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        return struct.pack("<B3x2I",
                           self._net_type,
                           self._ticks_per_interval,
                           self._global_max_ticks,
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
            update_function = self._stg_update_function
        else:
            update_function = self._update_function

        # set the number of examples to use in this stage
        if self._stg_examples is not None:
            num_examples = self._stg_examples
        else:
            num_examples = self._ex_set.num_examples

        # set the number of epochs to run in this stage
        if self._stg_epochs is not None:
            num_epochs = self._stg_epochs
        else:
            num_epochs = self._num_updates

        return struct.pack("<4B2I",
                           self._stage_id,
                           self.training,
                           update_function.value,
                           self._stg_reset,
                           num_examples,
                           num_epochs
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
            print ("error: pre-link and post-link groups required")
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

                # retrieve recorded tick_data from first output subgroup
                g = self.out_grps[0]
                ftv = g.t_vertex[0]
                rec_tick_data = ftv.read (
                    gfe.placements().get_placement_of_vertex (ftv),
                    gfe.buffer_manager(), MLPExtraRecordings.TICK_DATA.value
                    )

                # retrieve recorded outputs from every output group
                rec_outputs = [None] * len (self.out_grps)
                for g in self.out_grps:
                    rec_outputs[g.write_blk] = []
                    # append all subgroups together
                    for s in range (g.subgroups):
                        gtv = g.t_vertex[s]
                        rec_outputs[g.write_blk].append (gtv.read (
                            gfe.placements().get_placement_of_vertex (gtv),
                            gfe.buffer_manager(),
                            MLPVarSizeRecordings.OUTPUTS.value)
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

                # prepare to retrieve recorded data
                TICK_DATA_FORMAT = "<4I"
                TICK_DATA_SIZE = struct.calcsize(TICK_DATA_FORMAT)

                TOTAL_TICKS = len (rec_tick_data) // TICK_DATA_SIZE

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
                        outputs = []
                        # get tick outputs for each subgroup
                        for sg, rec_outs in enumerate (rec_outputs[g.write_blk]):
                            outputs += struct.unpack_from (
                                f"<{g.subunits[sg]}H",
                                rec_outs,
                                tk * struct.calcsize(f"<{g.subunits[sg]}H")
                                )

                        # print outputs
                        f.write (f"{g.units} 1\n")
                        tinx = tgt_inx * g.units
                        for u in range (g.units):
                            # outputs are s16.15 fixed-point numbers
                            out = (1.0 * outputs[u]) / (1.0 * (1 << 15))
                            t = g.targets[tinx + u]
                            if (t is None) or (t == float ('nan')):
                                tgt = "-"
                            else:
                                tgt = int (t)
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

            # retrieve recorded test results from last output subgroup
            g = self.out_grps[-1]
            ltv = g.t_vertex[g.subgroups - 1]
            rec_test_results = ltv.read (
                gfe.placements().get_placement_of_vertex (ltv),
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

        # compute number of subgroups
        for grp in self.groups:
            self.subgroups += grp.subgroups

        # create weight, sum, input and threshold
        # machine vertices associated with every subgroup
        for grp in self.groups:
            for sgrp in range (grp.subgroups):
                # create one weight core for every
                # (from_group/from_subgroup, group/subgroup) pair
                #TODO: all-zero cores can be optimised out
                wvs = []
                for from_grp in self.groups:
                    for from_sgrp in range (from_grp.subgroups):
                        wv = WeightVertex (self, grp, sgrp,
                                           from_grp, from_sgrp)
                        gfe.add_machine_vertex_instance (wv)
                        wvs.append (wv)
                grp.w_vertices.append (wvs)

                # create one sum core per subgroup
                sv = SumVertex (self, grp, sgrp)
                grp.s_vertex.append (sv)
                gfe.add_machine_vertex_instance (sv)

                # create one input core per subgroup
                iv = InputVertex (self, grp, sgrp)
                grp.i_vertex.append (iv)
                gfe.add_machine_vertex_instance (iv)

                # create one threshold core per subgroup
                tv = ThresholdVertex (self, grp, sgrp)
                grp.t_vertex.append (tv)
                gfe.add_machine_vertex_instance (tv)

        # groups and subgroups with special functions
        first_lds_grp = self.groups[0]
        first_subgroup_s_vertex = first_lds_grp.s_vertex[0]

        last_out_grp = self.output_chain[-1]
        last_out_subgroup_t_vertex = (
            last_out_grp.t_vertex[last_out_grp.subgroups - 1]
            )

        # create associated forward, backprop, link delta summation,
        # criterion, stop and sync machine edges for every subgroup
        for grp in self.groups:
            for sgrp in range (grp.subgroups):
                sv = grp.s_vertex[sgrp]
                iv = grp.i_vertex[sgrp]
                tv = grp.t_vertex[sgrp]

                for wv in grp.w_vertices[sgrp]:
                    from_grp  = wv.from_group
                    from_sgrp = wv.from_subgroup

                    from_sv = from_grp.s_vertex[from_sgrp]
                    from_tv = from_grp.t_vertex[from_sgrp]

                    # forward w to s link
                    gfe.add_machine_edge_instance (
                        MachineEdge (wv, sv),
                        wv.fwd_link
                        )

                    # forward t to w (multicast) link
                    gfe.add_machine_edge_instance (
                        MachineEdge (from_tv, wv),
                        from_tv.fwd_link
                        )

                    # backprop w to s link
                    gfe.add_machine_edge_instance (
                        MachineEdge (wv, from_sv),
                        wv.bkp_link
                        )

                    # backprop i to w (multicast) link
                    gfe.add_machine_edge_instance (
                        MachineEdge (iv, wv),
                        iv.bkp_link
                        )

                    # link delta summation w to s link
                    gfe.add_machine_edge_instance (
                        MachineEdge (wv, sv),
                        wv.lds_link
                        )

                    # link delta result (first group) s to w (multicast) link
                    gfe.add_machine_edge_instance (
                        MachineEdge (first_subgroup_s_vertex, wv),
                        first_subgroup_s_vertex.lds_link
                        )

                    # stop (last output group/subgroup) t to w (multicast) link
                    gfe.add_machine_edge_instance (
                        MachineEdge (last_out_subgroup_t_vertex, wv),
                        last_out_subgroup_t_vertex.stp_link
                        )

                    # intra-subgroup sync s to w (multicast) link
                    gfe.add_machine_edge_instance (
                        MachineEdge (sv, wv),
                        sv.fds_link
                        )

                    # inter-subgroup sync s to w (multicast) link
                    #NOTE: avoid duplicates
                    if grp != from_grp or sgrp != from_sgrp:
                        gfe.add_machine_edge_instance (
                            MachineEdge (from_sv, wv),
                            from_sv.fds_link
                            )

                # forward s to i link
                gfe.add_machine_edge_instance (
                    MachineEdge (sv, iv),
                    sv.fwd_link
                    )

                # forward i to t link
                gfe.add_machine_edge_instance (
                    MachineEdge (iv, tv),
                    iv.fwd_link
                    )

                # backprop t to i link
                gfe.add_machine_edge_instance (
                    MachineEdge (tv, iv),
                    tv.bkp_link
                    )

                # backprop s to t link
                gfe.add_machine_edge_instance (
                    MachineEdge (sv, tv),
                    sv.bkp_link
                    )

                # link delta summation s to s link
                if sgrp != 0:
                    # first subgroup collects from all other subgroups
                    gfe.add_machine_edge_instance (
                        MachineEdge (sv, grp.s_vertex[0]),
                        sv.lds_link
                        )
                elif grp != first_lds_grp:
                    # first group collects from all other groups
                    gfe.add_machine_edge_instance (
                        MachineEdge (sv, first_subgroup_s_vertex),
                        sv.lds_link
                        )

                # (output groups) t to t criterion link 
                if grp in self.output_chain:
                    # intra-group criterion link
                    if sgrp < (grp.subgroups - 1):
                        gfe.add_machine_edge_instance (
                            MachineEdge (tv, grp.t_vertex[sgrp + 1]),
                            tv.stp_link
                            )
                    elif grp != last_out_grp:
                        # inter-group criterion link
                        ngi = self.output_chain.index(grp) + 1
                        next_grp = self.output_chain[ngi]
                        gfe.add_machine_edge_instance (
                            MachineEdge (tv, next_grp.t_vertex[0]),
                            tv.stp_link
                            )

                # stop (last output group/subgroup) t to s (multicast) link
                gfe.add_machine_edge_instance (
                    MachineEdge (last_out_subgroup_t_vertex, sv),
                    last_out_subgroup_t_vertex.stp_link
                    )

                # stop (last output group/subgroup) t to i (multicast) link
                gfe.add_machine_edge_instance (
                    MachineEdge (last_out_subgroup_t_vertex, iv),
                    last_out_subgroup_t_vertex.stp_link
                    )

                # stop (last output group/subgroup) t to t (multicast) link
                if tv != last_out_subgroup_t_vertex:
                    gfe.add_machine_edge_instance (
                        MachineEdge (last_out_subgroup_t_vertex, tv),
                        last_out_subgroup_t_vertex.stp_link
                        )

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
