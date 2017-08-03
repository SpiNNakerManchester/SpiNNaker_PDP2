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
from mlp_examples import MLPExampleSet, MLPExample, MLPEvent


class MLPNetwork():
    """ top-level MLP network object.
            contains groups and links
            and top-level properties.
    """

    def __init__(self,
                net_type,
                intervals = 1,
                ticks_per_interval = None,
                ):
        """
        """
        # assign network parameter values from arguments
        self._net_type           = net_type.value
        self._ticks_per_interval = ticks_per_interval
        self._global_max_ticks   = (intervals * ticks_per_interval) + 1

        # default network parameter values
        self._train_group_crit = None
        self._test_group_crit  = None
        self._timeout          = MLPConstants.DEF_TIMEOUT
        self._num_epochs       = MLPConstants.DEF_NUM_EPOCHS

        # initialise lists of groups and links
        self.groups = []
        self.links  = []

        # OUTPUT groups form chain for convergence decision
        self._output_chain = []

        # track if initial weights have been loaded
        self._weights_loaded = False
        self._weights_file = None

        # keep track if examples have been loaded
        self._examples_loaded = False

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

        if (group_type == MLPGroupTypes.OUTPUT):
            _write_blk = len (self.output_chain)
            if len (self.output_chain):
                _is_first_out = 0
            else:
                _is_first_out = 1
        else:
            _write_blk    = 0
            _is_first_out = 0

        _group = MLPGroup (_id,
                           units        = units,
                           gtype        = group_type,
                           input_funcs  = input_funcs,
                           output_funcs = output_funcs,
                           write_blk    = _write_blk,
                           is_first_out = _is_first_out,
                           label        = label
                           )

        print "adding group {} [total: {}]".format (
                label,
                len (self.groups) + 1
                )

        self.groups.append (_group)

        if (group_type == MLPGroupTypes.OUTPUT):
            self.output_chain.append (_group)

        # OUTPUT and HIDDEN groups instantiate BIAS links by default
        if (group_type == MLPGroupTypes.OUTPUT or\
            group_type == MLPGroupTypes.HIDDEN):
            self.link (self.bias_group, _group)

        # initial weights file must be re-loaded!
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

        _link = MLPLink (pre_link_group  = pre_link_group,
                         post_link_group = post_link_group,
                         label           = _label
                         )

        print "adding link from {} to {} [total: {}]".format (\
            pre_link_group.label,
            post_link_group.label,
            len (self.links) + 1
            )

        self.links.append (_link)

        # initial weights file must be re-loaded!
        self._weights_loaded = False

        return _link


    def set (self,
             num_updates      = None,
             train_group_crit = None,
             test_group_crit  = None
             ):
        """ set a network parameter

        :param num_updates: number of training epochs to be done
        :param train_group_crit: criterion used to stop training
        :param test_group_crit: criterion used to stop testing

        :type num_updates: unsigned integer
        :type train_group_crit: s16.15 fixed-point value
        """
        if num_updates is not None:
            self._num_epochs = num_updates

        if train_group_crit is not None:
            self._train_group_crit = int (train_group_crit * (1 << 15))

        if test_group_crit is not None:
            self._test_group_crit = int (test_group_crit * (1 << 15))


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
            return

        print "reading Lens-style weights file"

        # compute the number of expected weights in the file
        _num_wts = 0
        for to_grp in self.groups:
            for frm_grp in to_grp.links_from:
                _num_wts = _num_wts + to_grp.units * frm_grp.units

        # check that it is the correct file type
        _wf = open (self._weights_file, "r")

        if int (_wf.readline ()) != MLPConstants.MAGIC_LENS_WEIGHT_COOKIE:
            print "error: incorrect weights file type"
            _wf.close ()
            return

        # check that the file contains the right number of weights
        if int (_wf.readline ()) != _num_wts:
            print "error: incorrect number of weights in file"
            _wf.close ()
            return

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


    def read_Lens_examples_file (self,
                                 examples_file
                                 ):
        """ reads a Lens-style examples file

        Lens online manual:
            http://web.stanford.edu/group/mbc/LENSManual/

        File format:

        proc:  <S set-proc>
        max:   <R set-maxTime>
        min:   <R set-minTime>
        grace: <R set-graceTime>
        defI:  <R set-defaultInput>
        actI:  <R set-activeInput>
        defT:  <R set-defaultTarget>
        actT:  <R set-activeTarget>
        ;

        for each example:
          name:   <S example-name>
          proc:   <S example-proc>
          freq:   <R example-frequency>
          <I example-numEvents>   this can be left out if it is 1

          for each list of events:
            [(<I event> | <I event>-<I event> | *)
              proc:  <S event-proc>
              max:   <R event-maxTime>
              min:   <R event-minTime>
              grace: <R event-graceTime>
              defI:  <R event-defaultInput>
              actI:  <R event-activeInput>
              defT:  <R event-defaultTarget>
              actT:  <R event-activeTarget>
            ]

            (I:|i:|T:|t:|B:|b:|) (
              dense range:  (<S group-name> <I first-unit>) (<R input-value>) |
              sparse range: {<S group-name> <R input-value>} [* | (<I unit> | <I unit>-<I unit>)]
            )
          ;
        """
        # check if file exists
        if os.path.isfile (examples_file):
            self._examples_file = examples_file
        elif os.path.isfile ("data/{}".format (examples_file)):
            self._examples_file = "data/{}".format (examples_file)
        else:
            self._examples_file = None
            print "error: cannot open examples file: {}".\
                format (examples_file)
            return

        print "reading Lens-style examples file"

        _ef = open (self._examples_file, "r")

        # create new example set
        _set = MLPExampleSet ()

        # process example file header
        _line = _ef.readline ()
        while (';' not in _line):
            if ('proc:' in _line):
                print "set procedure not supported"
            elif ('max:' in _line):
                _, _val = _line.split(':')
                _set.max_time = float (_val)
            elif ('min:' in _line):
                _, _val = _line.split(':')
                _set.min_time = float (_val)
            elif ('grace:' in _line):
                _, _val = _line.split(':')
                _set.grace_time = float (_val)
            elif ('defI:' in _line):
                _, _val = _line.split(':')
                _set.def_input = float (_val)
            elif ('actI:' in _line):
                print "set active input not supported"
            elif ('defT:' in _line):
                _, _val = _line.split(':')
                _set.def_target = float (_val)
            elif ('actT:' in _line):
                print "set active target not supported"
            else:
                # ';' is optional
                break

            _line = _ef.readline ()

        while (_line != ""):
            _line = _ef.readline ()

        # process each example in the set
        _ex_id = 0
        while (_line != ""):
            # create new example
            _ex = MLPExample (_ex_id)

            # process the example header
            _done = False
            while not _done:
                _done = True
                try:
                    if ('proc:' in _line):
                        print "example procedure not supported"
                    elif ('freq:' in _line):
                        print "example frequency not supported"
                    elif ('name:' in _line):
                        _, _ex.name = _line.split(':')
                        _line = _ef.readline ()
                    else:
                        # try to get non-default number of events
                        _num_ev = int (_line)
                        _done = True

                    # process next line
                    _line = _ef.readline ()
                    if (_line == ""):
                        print "unexpected end-of-file - read aborted"
                        _ef.close ()
                        return None
                except:
                    # if absent number of events defaults to 1
                    _num_ev = 1
                    _done = True
                    print "default num_ev!"
                    print _line

            # process each event in the example
            _ev_id = 0
            while (False):
                # create new event
                _ev = MLPEvent (_ev_id)


                # add new event to example event list
                _ex.events.append (_ev)

                # prepare for next example
                _ex_id = _ex_id + 1
                _line = _ef.readline ()

            # add new example to set example list
            _set.examples.append (_ex)

            # prepare for next example
            _ex_id = _ex_id + 1

        # clean up
        _ef.close ()

        # mark examples file as loaded
        self._examples_loaded = True

        return _set


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

                # create backprop w to s links
                g.add_machine_edge_instance (MachineEdge (w, _frmg.s_vertex),
                                             w.bkp_link)

                # create forward synchronisation w to t links
                g.add_machine_edge_instance (MachineEdge (w, _frmg.t_vertex),
                                             w.fds_link)

                # create backprop i to w (multicast) links
                g.add_machine_edge_instance (MachineEdge (grp.i_vertex, w),
                                             grp.i_vertex.bkp_link)

                # create forward t to w (multicast) links
                g.add_machine_edge_instance (MachineEdge (_frmg.t_vertex, w),
                                             _frmg.t_vertex.fwd_link)

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


    def train (self,
               num_examples = None
               ):
        """ train the application graph for a number of epochs

        :param num_examples: number of examples from the set to be used

        :type  num_examples: unsigned integer
        """
        self._training     = 1
        self._num_examples = num_examples

        # run the application
        self.run ()


    def test (self,
              num_examples = None
              ):
        """ test the application graph without training

        :param num_examples: number of examples from the set to be used

        :type  num_examples: unsigned integer
        """
        self._training     = 0
        self._num_examples = num_examples

        # run the application
        self.run ()


    def run (self):
        """ run the application graph
        """
        # cannot run unless examples have been loaded
        if not self._examples_loaded:
            print "run aborted: examples not loaded"
            return

        # cannot run unless weights file exists
        if self._weights_file is None:
            print "run aborted: weights file not given"
            return

        # may need to reload initial weights file if
        # application graph was modified after first load
        if self._weights_file is not None and\
            not self._weights_loaded:
            self.read_Lens_weights_file (self._weights_file)

        # generate machine graph
        self.generate_machine_graph ()

        # run application based on the machine graph
        g.run (None)

        # wait for the application to finish
        print "running: waiting for application to finish"
        _txrx = g.transceiver()
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
        raw_input ('paused for debug: press enter to exit')

        print "exit: application finished"
        # let the gfe clean up
        #g.stop()
