import os
import struct
import time

import spinnaker_graph_front_end as g

from pacman.model.graphs.machine import MachineEdge

from spinnman.model.enums.cpu_state import CPUState

from spinn_front_end_common.utilities import globals_variables

from input_vertex     import InputVertex
from sum_vertex       import SumVertex
from threshold_vertex import ThresholdVertex
from weight_vertex    import WeightVertex

from mlp_types    import MLPGroupTypes, MLPConstants
from mlp_group    import MLPGroup
from mlp_link     import MLPLink
from mlp_examples import MLPExampleSet


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
        self._ticks_per_interval = ticks_per_interval

        # default network parameter values
        self._global_max_ticks = (intervals * ticks_per_interval) + 1
        self._train_group_crit = None
        self._test_group_crit  = None
        self._timeout          = MLPConstants.DEF_TIMEOUT
        self._num_epochs       = MLPConstants.DEF_NUM_EPOCHS

        # initialise lists of groups and links
        self.groups = []
        self.links  = []

        # initialise lists of INPUT and OUTPUT groups
        self.in_grps  = []
        self.out_grps = []

        # OUTPUT groups form a chain for convergence decision
        self._output_chain = []

        # track if initial weights have been loaded
        self._weights_loaded = False
        self._weights_file = None

        # initialise example set
        self._ex_set = None

        # create single-unit Bias group by default
        self._bias_group = self.group (units        = 1,
                                       group_type   = MLPGroupTypes.BIAS,
                                       label        = "Bias"
                                       )


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
    def num_write_blocks (self):
        return self._num_write_blks

    @property
    def timeout (self):
        return self._timeout

    @property
    def output_chain (self):
        return self._output_chain

    @property
    def bias_group (self):
        return self._bias_group

    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) network_conf in mlp_types.h:

            typedef struct network_conf
            {
              uchar net_type;
              uchar training;
              uint  num_epochs;
              uint  num_examples;
              uint  ticks_per_int;
              uint  global_max_ticks;
              uint  num_write_blks;
              uint  timeout;
            } network_conf_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        return struct.pack("<2B2x6I",
                           self._net_type,
                           self._training,
                           self._num_epochs,
                           self._num_examples,
                           self._ticks_per_interval,
                           self._global_max_ticks,
                           self._num_write_blks,
                           self._timeout
                           )


    def group (self,
               units        = None,
               group_type   = MLPGroupTypes.HIDDEN,
               input_funcs  = None,
               output_funcs = None,
               label        = None
               ):
        """ add a group to the network

        :param units: number of units that form the group
        :param group_type: Lens-style group type
        :param input_funcs: functions applied in the input pipeline
        :param output_funcs: functions appllied in the output pipeline
        :param label: human-readable group identifier

        :type units: unsigned integer
        :type group_type: enum MLPGroupTypes
        :type input_funcs: enum MLPInputProcs
        :type output_funcs: enum MLPOutputProcs
        :type label: string

        :return: a new group object
        """
        _id = len (self.groups)

        # set properties for OUTPUT group
        if (group_type == MLPGroupTypes.OUTPUT):
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

        print "adding group {} [total: {}]".format (
                label,
                len (self.groups)
                )

        # if it's an INPUT group add to list
        if (group_type == MLPGroupTypes.INPUT):
            self.in_grps.append (_group)

        # if it's an OUTPUT group add to list and to the tail of the chain
        if (group_type == MLPGroupTypes.OUTPUT):
            self.out_grps.append (_group)
            self.output_chain.append (_group)

        # OUTPUT and HIDDEN groups instantiate BIAS links by default
        if (group_type == MLPGroupTypes.OUTPUT or\
            group_type == MLPGroupTypes.HIDDEN):
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
        if label is None:
            _label = "{}-{}".format (pre_link_group.label,
                                     post_link_group.label
                                     )
        else:
            _label = label

        # check that enough data is provided
        if (pre_link_group is None) or (post_link_group is None):
            print ("error: pre and post link groups required")
            return None

        # instantiate a new link
        _link = MLPLink (pre_link_group  = pre_link_group,
                         post_link_group = post_link_group,
                         label           = _label
                         )

        # add new link to the network list
        self.links.append (_link)

        print "adding link from {} to {} [total: {}]".format (\
            pre_link_group.label,
            post_link_group.label,
            len (self.links)
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
        _set = MLPExampleSet (label = label)

        # add example set to the network list
        self._ex_set = _set

        print "adding example set {}".format (label)

        return _set


    def set (self,
             num_updates      = None,
             train_group_crit = None,
             test_group_crit  = None
             ):
        """ set a network parameter to the given value

        :param num_updates: number of training epochs to be done
        :param train_group_crit: criterion used to stop training
        :param test_group_crit: criterion used to stop testing

        :type num_updates: unsigned integer
        :type train_group_crit: float
        """
        if num_updates is not None:
            print "setting num_epochs to {}".format (num_updates)
            self._num_epochs = num_updates

        if train_group_crit is not None:
            print "setting train_group_crit to {}".format (train_group_crit)
            self._train_group_crit = train_group_crit

        if test_group_crit is not None:
            print "setting test_group_crit to {}".format (test_group_crit)
            self._test_group_crit = test_group_crit


    def read_Lens_weights_file (self,
                                weights_file
                                ):
        """ reads a Lens-style weights file

        Lens online manual:
            http://web.stanford.edu/group/mbc/LENSManual/

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
            print "error: cannot open weights file: {}".\
                format (weights_file)
            return False

        print "reading Lens-style weights file"

        # compute the number of expected weights in the file
        _num_wts = 0
        for to_grp in self.groups:
            for frm_grp in to_grp.links_from:
                _num_wts = _num_wts + to_grp.units * frm_grp.units

        # check that it is the correct file type
        _wf = open (self._weights_file, "r")

        if int (_wf.readline ()) != MLPConstants.LENS_WEIGHT_MAGIC_COOKIE:
            print "error: incorrect weights file type"
            _wf.close ()
            return False

        # check that the file contains the right number of weights
        if int (_wf.readline ()) != _num_wts:
            print "error: incorrect number of weights in file"
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


    def generate_machine_graph (self):
        """ generates a machine graph for the application graph
        """
        print "generating machine graph"

        # setup the machine graph
        g.setup ()

        # set the number of write blocks before generating vertices
        self._num_write_blks = len (self.output_chain)

        # create associated weight, sum, input and threshold
        # machine vertices for every network group
        for grp in self.groups:
            # create one weight core per (from_group, group) pair
            # NOTE: all-zero cores can be optimised out
            for from_grp in self.groups:
                wv = WeightVertex (self, grp, from_grp)
                grp.w_vertices.append (wv)
                g.add_machine_vertex_instance (wv)

            # create one sum core per group
            sv = SumVertex (self, grp)
            grp.s_vertex = sv
            g.add_machine_vertex_instance (sv)

            # create one input core per group
            iv = InputVertex (self, grp)
            grp.i_vertex = iv
            g.add_machine_vertex_instance (iv)

            # create one sum core per group
            tv = ThresholdVertex (self, grp)
            grp.t_vertex = tv
            g.add_machine_vertex_instance (tv)

        # create associated forward, backprop, synchronisation and
        # stop machine edges for every network group
        for grp in self.groups:
            for w in grp.w_vertices:
                _frmg = w.from_group

                # create forward w to s links
                g.add_machine_edge_instance (MachineEdge (w, grp.s_vertex),
                                             w.fwd_link)

                # create forward t to w (multicast) links
                g.add_machine_edge_instance (MachineEdge (_frmg.t_vertex, w),
                                             _frmg.t_vertex.fwd_link)

                # create backprop w to s links
                g.add_machine_edge_instance (MachineEdge (w, _frmg.s_vertex),
                                             w.bkp_link)

                # create backprop i to w (multicast) links
                g.add_machine_edge_instance (MachineEdge (grp.i_vertex, w),
                                             grp.i_vertex.bkp_link)

                # create forward synchronisation w to t links
                g.add_machine_edge_instance (MachineEdge (w, _frmg.t_vertex),
                                             w.fds_link)

            # create forward s to i link
            g.add_machine_edge_instance (MachineEdge (grp.s_vertex,
                                                      grp.i_vertex),
                                         grp.s_vertex.fwd_link)

            # create backprop s to t link
            g.add_machine_edge_instance (MachineEdge (grp.s_vertex,
                                                      grp.t_vertex),
                                         grp.s_vertex.bkp_link)

            # create forward i to t link
            g.add_machine_edge_instance (MachineEdge (grp.i_vertex,
                                                      grp.t_vertex),
                                         grp.i_vertex.fwd_link)

            # create backprop t to i link
            g.add_machine_edge_instance (MachineEdge (grp.t_vertex,
                                                      grp.i_vertex),
                                         grp.t_vertex.bkp_link)

            # create stop links, if OUTPUT group
            if grp in self.output_chain:
                # if last OUTPUT group broadcast stop decision
                if grp == self.output_chain[-1]:
                    for stpg in self.groups:
                        # create stop links to all w cores
                        for w in stpg.w_vertices:
                            g.add_machine_edge_instance\
                              (MachineEdge (grp.t_vertex, w),
                               grp.t_vertex.stp_link)

                        # create stop links to all s cores
                        g.add_machine_edge_instance\
                         (MachineEdge (grp.t_vertex, stpg.s_vertex),\
                          grp.t_vertex.stp_link)

                        # create stop links to all i cores
                        g.add_machine_edge_instance\
                         (MachineEdge (grp.t_vertex, stpg.i_vertex),\
                          grp.t_vertex.stp_link)

                        # create stop links to t cores (no link to itself!)
                        if stpg != grp:
                            g.add_machine_edge_instance\
                             (MachineEdge (grp.t_vertex, stpg.t_vertex),\
                              grp.t_vertex.stp_link)
                else:
                    # create stop link to next OUTPUT group in chain
                    _inx  = self.output_chain.index (grp)
                    _stpg = self.output_chain[_inx + 1]
                    g.add_machine_edge_instance (MachineEdge (grp.t_vertex,
                                                              _stpg.t_vertex),
                                                 grp.t_vertex.stp_link)


    def train (self):
        """ train the application graph
        """
        self._training = 1

        # run the application
        self.run ()


    def test (self):
        """ test the application graph without training
        """
        self._training = 0

        # run the application
        self.run ()


    def run (self):
        """ run the application graph
        """
        # cannot run unless weights file exists
        if self._weights_file is None:
            print "run aborted: weights file not given"
            return

        # may need to reload initial weights file if
        # application graph was modified after load
        if not self._weights_loaded:
            if not self.read_Lens_weights_file (self._weights_file):
                print "run aborted: error reading weights file"

        # cannot run unless example set exists
        if self._ex_set is None:
            print "run aborted: no example set"
            return

        # cannot run unless examples have been loaded
        if not self._ex_set.examples_loaded:
            print "run aborted: examples not loaded"
            return

        # generate summary set, example and event data
        self._num_examples = self._ex_set.compile (self)
        if self._num_examples == 0:
            print "run aborted: error compiling example set"
            return

        # generate machine graph
        self.generate_machine_graph ()

        # run application based on the machine graph
        g.run (None)

        # wait for the application to finish
        print "running: waiting for application to finish"
        _txrx = g.transceiver ()
        _app_id = globals_variables.get_simulator ()._app_id
        _running = _txrx.get_core_state_count (_app_id, CPUState.RUNNING)
        while _running > 0:
            time.sleep (0.5)
            _error = _txrx.get_core_state_count\
                    (_app_id, CPUState.RUN_TIME_EXCEPTION)
            _wdog = _txrx.get_core_state_count (_app_id, CPUState.WATCHDOG)
            if _error > 0 or _wdog > 0:
                print "application stopped: cores failed ({}\
                     RTE, {} WDOG)".format (_error, _wdog)
                break
            _running = _txrx.get_core_state_count (_app_id, CPUState.RUNNING)


    def end (self):
        """ clean up before exiting
        """
        # pause to allow debugging
        raw_input ('paused: press enter to exit')

        print "exit: application finished"
        # let the gfe clean up
        #g.stop()
