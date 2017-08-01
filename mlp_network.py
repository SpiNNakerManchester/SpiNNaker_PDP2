import os
import struct

import spinnaker_graph_front_end as g

from pacman.model.graphs.machine import MachineEdge

from input_vertex     import InputVertex
from sum_vertex       import SumVertex
from threshold_vertex import ThresholdVertex
from weight_vertex    import WeightVertex

from mlp_group import MLPGroup
from mlp_link  import MLPLink
from mlp_types import MLPGroupTypes, MLPConstants


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
        # assign network parameter initial values
        self._net_type           = net_type.value
        self._ticks_per_interval = ticks_per_interval
        self._global_max_ticks   = (intervals * ticks_per_interval) + 1
        self._timeout            = 10000

        # initialise lists of groups and links
        self.groups = []
        self.links  = []

        # OUTPUT groups form chain for convergence decision
        self._output_chain = []

        # keep track if initial weights have been loaded
        self._initial_weights_loaded = 0
        self._initial_weights_file = None

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

            pack: standard sizes, little-endian byte-order,
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

        :param units:
        :param group_type:
        :param label:

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
        self._initial_weights_loaded = 0

        return _group


    def link (self,
              pre_link_group = None,
              post_link_group = None,
              label = None
              ):
        """ add a link to the network

        :param pre_link_group: link source group
        :param post_link_group: link destination group

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
        self._initial_weights_loaded = 0

        return _link


    def read_Lens_weights_file (self,
                                weights_file
                                ):
        """ reads a Lens-style weights file

        File format (Lens online manual):

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
            self.weights_file = weights_file
        elif os.path.isfile ("data/{}".format (weights_file)):
            self.weights_file = "data/{}".format (weights_file)
        else:
            self.weights_file = None
            print "error: cannot open weights file: {}".\
                format (self.initial_weights_file)
            return

        print "reading Lens-style weights file"

        # compute the number of expected weights in the file
        _num_wts = 0
        for to_grp in self.groups:
            for frm_grp in to_grp.links_from:
                _num_wts = _num_wts + to_grp.units * frm_grp.units

        # check that it is the correct file type
        _wf = open (self.weights_file, "r")

        if int (_wf.readline()) != MLPConstants.MAGIC_LENS_WEIGHT_COOKIE:
            print "error: incorrect weights file type"
            _wf.close ()
            return

        # check that the file contains the right number of weights
        if int (_wf.readline()) != _num_wts:
            print "error: incorrect number of weights in file"
            _wf.close ()
            return

        # read weights from file and store them in the corresponding group
        _num_values = int (_wf.readline())
        _ = _wf.readline()  # discard number of updates

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
                            grp.weights[fgrp].append(float (_wf.readline()))
                            if _num_values >= 2:
                                _ = _wf.readline()  # discard
                            if _num_values >= 3:
                                _ = _wf.readline()  # discard

        # clean up
        _wf.close ()

        # mark weights file as loaded
        self._initial_weights_loaded = 1


    def generate_machine_graph (self,
                                ):
        """ generates a machine graph for simulation
        """
        print "generating machine graph"

        # setup the machine graph
        g.setup ()

        # set the number of write blocks before generating vertices
        self._num_write_blks = len (self.output_chain)

        # create associated weight, sum, input and threshold
        # machine vertices for every network group
        _num_groups = len (self.groups)

        for grp in self.groups:
            # create one weight core per network group, including this one
            # NOTE: could be optimised. If so, see NOTEs below.
            for fgrp in self.groups:
                wv = WeightVertex (self,
                                   grp,
                                   from_group = fgrp
                                   )

                grp.w_vertices.append (wv)
                g.add_machine_vertex_instance (wv)

            # NOTE: if all-zero w cores are optimised out
            # then fwd_expect and bkp_expect must be adjusted
            sv = SumVertex (self,
                            grp,
                            fwd_expect = _num_groups,
                            bkp_expect = _num_groups
                            )

            grp.s_vertex = sv
            g.add_machine_vertex_instance (sv)

            iv = InputVertex (self,
                              grp
                              )

            grp.i_vertex = iv
            g.add_machine_vertex_instance (iv)

            # check if last output group in daisy chain
            if grp == self.output_chain[-1]:
                _is_last_out = 1
            else:
                _is_last_out = 0

            # NOTE: if all-zero w cores are optimised out
            # then fwd_sync_expecr must be adjusted
            tv = ThresholdVertex (self,
                                  grp,
                                  fwd_sync_expect = _num_groups,
                                  is_last_out     = _is_last_out
                                  )

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
                        for w in stpg.w_vertices:
                            g.add_machine_edge_instance\
                              (MachineEdge (grp.t_vertex, w),
                               grp.t_vertex.stp_link)

                        g.add_machine_edge_instance\
                         (MachineEdge (grp.t_vertex, stpg.s_vertex),\
                          grp.t_vertex.stp_link)

                        g.add_machine_edge_instance\
                         (MachineEdge (grp.t_vertex, stpg.i_vertex),\
                          grp.t_vertex.stp_link)

                        # no link to itself!
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
               num_updates  = None,
               num_examples = None
               ):
        """ train the application graph for a number of epochs

        :param num_epochs:
        :param num_examples:

        :type  num_epochs: int
        :type  num_examples: int
        """
        print "g.run ()"

        self._training     = 1
        self._num_epochs   = num_updates
        self._num_examples = num_examples

        # generate machine graph
        self.generate_machine_graph ()

        # run simulation of the machine graph
        g.run (None)


    def test (self,
               num_examples = None
              ):
        """ test the application graph without training

        :param num_examples:

        :type  num_examples: int
        """
        print "g.run ()"

        self._training     = 0
        self._num_examples = num_examples

        # may need to reload initial weights file if
        # application graph was modified after first load
        if self.weights_file is not None and\
            not self._initial_weights_loaded:
            self.read_Lens_weights_file(self.weights_file)

        # generate machine graph
        self.generate_machine_graph ()

        # run simulation of the machine graph
        g.run (None)


    def end (self):
        """ clean up before exiting
        """
        print "g.stop ()"
        #g.stop()
