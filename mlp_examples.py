import os
import struct
import re
from mlp_types import MLPConstants
from numpy.distutils.lib2def import DEFAULT_NM


class MLPExampleSet ():
    """ an MLP Lens-style example set
    """

    def __init__(self,
                 label       = None,
                 max_time    = None,
                 min_time    = None,
                 grace_time  = None,
                 def_input   = None,
                 def_target  = None,
                ):
        """
        """
        self.label      = label
        self.max_time   = max_time
        self.min_time   = min_time
        self.grace_time = grace_time
        self.def_input  = def_input
        self.def_target = def_target

        # start with an empty list of examples
        self.examples   = []

        # track if examples have been loaded
        self.examples_loaded = False

        print "creating example set"


    def set (self,
             max_time    = None,
             min_time    = None,
             grace_time  = None
             ):
        """ set a network parameter to the given value

        :param max_time: maximum time per example
        :param min_time: minimum time per example
        :param grace_time: initial time without evaluation of convergence

        :type max_time: float
        :type min_time: float
        :type grace_time: float
        """
        if max_time is not None:
            print "setting max_time to {}".format (max_time)
            self.max_time = max_time

        if min_time is not None:
            print "setting min_time to {}".format (min_time)
            self.min_time = min_time

        if grace_time is not None:
            print "setting grace_time to {}".format (grace_time)
            self.grace_time = grace_time


    @property
    def config (self):
        """ returns a packed string that corresponds to
            (C struct) mlp_set in mlp_types.h:

            typedef struct mlp_set
            {
              uint    num_examples;
              fpreal  max_time;
              fpreal  min_time;
              fpreal  grace_time;
            } mlp_set_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # max_time is represented in fixed-point s15.16 notation
        _max_time = int (self.max_time *\
                         (1 << MLPConstants.FPREAL_SHIFT))

        # min_time is represented in fixed-point s15.16 notation
        _min_time = int (self.min_time *\
                         (1 << MLPConstants.FPREAL_SHIFT))

        # grace_time is represented in fixed-point s15.16 notation
        _grace_time = int (self.grace_time *\
                           (1 << MLPConstants.FPREAL_SHIFT))

        return struct.pack("<4I",
                           len (self.examples),
                           _max_time,
                           _min_time,
                           _grace_time
                           )


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

        print "processing example set header"

        # process example set header
        _line = _ef.readline ()
        while (';' not in _line):
            if ('proc:' in _line):
                print "set procedure not supported"
            elif ('max:' in _line):
                _, _val = _line.split (':')
                try:
                    self.max_time = float (_val)
                except:
                    self.max_time = float ('nan')
                print "setting set max:{}".format (self.max_time)
            elif ('min:' in _line):
                _, _val = _line.split (':')
                try:
                    self.min_time = float (_val)
                except:
                    self.min_time = float ('nan')
                print "setting set min:{}".format (self.min_time)
            elif ('grace:' in _line):
                _, _val = _line.split (':')
                try:
                    self.grace_time = float (_val)
                except:
                    self.grace_time = float ('nan')
                print "setting set grace:{}".format (self.grace_time)
            elif ('defI:' in _line):
                _, _val = _line.split (':')
                try:
                    self.def_input = float (_val)
                except:
                    self.def_input = float ('nan')
                print "setting set defI:{}".format (self.def_input)
            elif ('actI:' in _line):
                print "set active input not supported"
            elif ('defT:' in _line):
                _, _val = _line.split (':')
                try:
                    self.def_target = float (_val)
                except:
                    self.def_target = float ('nan')
                print "setting set defT:{}".format (self.def_target)
            elif ('actT:' in _line):
                print "set active target not supported"
            else:
                # ';' is optional
                break

            _line = _ef.readline ()

        # ';' is optional
        if (';' in _line):
            _line = _ef.readline ()

        # process each example in the set
        _ex_id = 0
        while (_line != ""):
            # create new example, initially empty
            _ex = MLPExample (_ex_id)

            print "processing example {}".format (_ex_id)

            # process the example header
            _done = False
            while not _done:
                if ('name:' in _line):
                    _, _name = _line.split (':')
                    _ex.name = _name.rstrip ()
                    print "setting example name:{}".format (_ex.name)

                elif ('proc:' in _line):
                    print "example procedure not supported"

                elif ('freq:' in _line):
                    _, _freq = _line.split (':')
                    _ex.freq = float (_freq)
                    print "setting example freq:{}".format (_ex.freq)

                else:
                    # try to get number of events
                    try:
                        _num_ev = int (_line)
                        _done = True
                        print "setting example num_ev:{}".format (_num_ev)
                    except:
                        # if absent, number of events defaults to 1
                        _num_ev = 1
                        print "setting by default example num_ev:{}".format (_num_ev)
                        break

                # prepare to process next line
                _line = _ef.readline ()
                if (_line == ""):
                    print "error: unexpected end-of-file"
                    _ef.close ()
                    return None

            # process each event in the example
            # TODO: need to complete event list processing!
            _ev_id = -1
            while (';' not in _line):
                # read event list, if present
                if ("[" in _line):
                    _ev_list = _line.strip ()
                    while ("]" not in _line):
                        _line = _ef.readline ()
                        _ev_list = _ev_list + " " + _line.strip ()
                else:
                    _ev_list = None

                _ev_id += 1

                # create new event
                _ev = MLPEvent (_ev_id)

                print "reading event {}".format (_ev_id)

                # get inputs and targets for every event
                _isd = False
                _tsd = False
                while (not _isd or not _tsd):
                    # instantiate new list of values
                    _vl = MLPEventValues ()

                    # check line for inputs
                    if ('I:' in _line) or ('i:' in _line):
                        _, _is = _line.split (":")
                        for s in re.findall (r'\S+', _is):
                            if "(" in s:
                                _vl.name = s.strip ("()")
                            else:
                                _vl.values.append (float (s))

                        print "added inputs {}:{}".format (_vl.name, _vl.values)

                        _ev.inputs.append (_vl)
                        _isd = True

                    # check line for targets
                    elif ('T:' in _line) or ('t:' in _line):
                        _, _ts = _line.split (":")
                        for s in re.findall (r'\S+', _ts):
                            if "(" in s:
                                _vl.name = s.strip ("()")
                            else:
                                _vl.values.append (float (s.strip (";")))

                        print "added targets {}:{}".format (_vl.name, _vl.values)

                        _ev.targets.append (_vl)
                        _tsd = True

                    # check line for both inputs and targets
                    elif ('B:' in _line) or ('b:' in _line):
                        _, _bs = _line.split (":")
                        for s in re.findall\
                            ("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*",\
                             _bs):
                            _vl.values.append (float (s))

                        print "added inputs {}".format (_vl.values)
                        print "added targets {}".format (_vl.values)

                        _ev.inputs.append (_vl)
                        _ev.targets.append (_vl)
                        _isd = True
                        _tsd = True

                    # check if final event in example
                    if (";" in _line):
                        break

                    # process new line
                    _line = _ef.readline ()
                    if (_line == ""):
                        print "error: unexpected end-of-file"
                        _ef.close ()
                        return None

                # add new event to example event list
                _ex.events.append (_ev)

                # prepare for next example
                _ev_id = _ev_id + 1

            # check that all events were processed
            if (_ev_id != _num_ev):
                print "error: inconsistent number of events"
                _ef.close ()
                return None

            # add new example to set example list
            self.examples.append (_ex)

            # prepare for next example
            _ex_id = _ex_id + 1
            _line = _ef.readline ()

        # clean up
        _ef.close ()

        # mark examples file as loaded
        self.examples_loaded = True


    def compile (self, network):
        """ process an example set to produce summary set,
            example and event data. Additionally produce
            an input subset for each INPUT group and a
            target subset for each OUTPUT group.
        """
        self._net = network

        # process example set
        print "compiling {}".format (self.label)

        # create example set configuration
        self.set_config = self.config

        # create example and event configuration lists
        self.example_config = []
        self.event_config   = []

        _ev_idx = 0
        _it_idx = 0

        # create a configuration for every example
        for _n, _ex in enumerate (self.examples):
            self.example_config.append (_ex.config (_n, _ev_idx))
            _ev_idx = _ev_idx + len (_ex.events)

            # process every event in the example
            for _ev in _ex.events:
                # create an event configuration
                self.event_config.append (_ev.config (self, _it_idx))
                _it_idx = _it_idx + 1

                # set default input value for this event
                if _ev.def_input is not None:
                    _defi = _ev.def_input
                elif self.def_input is not None:
                    _defi = self.def_input
                else:
                    _defi = None

                # set default target value for this event
                if _ev.def_target is not None:
                    _deft = _ev.def_target
                elif self.def_target is not None:
                    _deft = self.def_target
                else:
                    _deft = None

                # process event inputs
                _grps_done = []
                for ei in _ev.inputs:
                    if ei.name is None:
                        if len (self._net.in_grps) == 1:
                            self._net.in_grps[0].inputs += ei.values
                            _grps_done.append(self._net.in_grps[0])
                        else:
                            print "error: not enough inputs in event"
                            return 0
                    else:
                        for g in self._net.in_grps:
                            if g.label == ei.name:
                                _grps_done.append (g)
                                g.inputs += ei.values
                                break

                # add default inputs for not-listed groups
                for g in self._net.in_grps:
                    if g not in _grps_done:
                        for _ in range (g.units):
                            g.inputs.append (_defi)

                    print "{}: {} inputs".format (g.label, len (g.inputs))

                # process event targets
                _grps_done = []
                for et in _ev.targets:
                    if et.name is None:
                        if len (self._net.out_grps) == 1:
                            self._net.out_grps[0].targets += et.values
                            _grps_done.append(self._net.out_grps[0])
                        else:
                            print "error: not enough targets in event"
                            return 0
                    else:
                        for g in self._net.out_grps:
                            if g.label == et.name:
                                _grps_done.append (g)
                                g.targets += et.values
                                break

                # add default targets for not-listed groups
                for g in self._net.out_grps:
                    if g not in _grps_done:
                        for _ in range (g.units):
                            g.targets.append (_deft)

                    print "{}: {} targets".format (g.label, len (g.targets))

        return len (self.examples)


class MLPExample ():
    """ an MLP Lens-style example
    """

    def __init__(self,
                 ex_id,
                 name = None,
                 freq = None
                 ):
        """
        """
        self.id   = ex_id
        self.name = name
        self.freq = freq

        # start with an empty list of events
        self.events = []

        print "creating example {}".format (ex_id)


    def config (self, num, ev_idx):
        """ returns a packed string that corresponds to
            (C struct) mlp_example in mlp_types.h:

            typedef struct mlp_example
            {
              uint   num;
              uint   num_events;
              uint   ev_idx;
              fpreal freq;
            } mlp_example_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # freq is represented in fixed-point s15.16 notation
        if self.freq is not None:
            _freq = int (self.freq * (1 << MLPConstants.FPREAL_SHIFT))
        else:
            _freq = int (MLPConstants.DEF_EX_FREQ *\
                         (1 << MLPConstants.FPREAL_SHIFT))

        return struct.pack("<4I",
                           num,
                           len (self.events),
                           ev_idx,
                           _freq
                           )


class MLPEvent ():
    """ an MLP Lens-style event
    """

    def __init__(self,
                 ev_id,
                 max_time    = None,
                 min_time    = None,
                 grace_time  = None,
                 def_input   = None,
                 def_target  = None
                 ):
        """
        """
        self.id         = ev_id
        self.max_time   = max_time
        self.min_time   = min_time
        self.grace_time = grace_time
        self.def_input  = def_input
        self.def_target = def_target

        # start with empty lists of inputs and targets
        self.inputs  = []
        self.targets = []

        print "creating event {}".format (ev_id)


    def config (self, ex_set, it_idx):
        """ returns a packed string that corresponds to
            (C struct) mlp_event in mlp_types.h:

            typedef struct mlp_event
            {
              fpreal  max_time;
              fpreal  min_time;
              fpreal  grace_time;
              uint    it_idx;
            } mlp_event_t;

            pack: standard sizes, little-endian byte order,
            explicit padding
        """
        # max_time is represented in fixed-point s15.16 notation
        if self.max_time is not None:
            _max_time = int (self.max_time *\
                             (1 << MLPConstants.FPREAL_SHIFT))
        else:
            _max_time = MLPConstants.FPREAL_NaN

        # min_time is represented in fixed-point s15.16 notation
        if self.min_time is not None:
            _min_time = int (self.min_time *\
                             (1 << MLPConstants.FPREAL_SHIFT))
        else:
            _min_time = MLPConstants.FPREAL_NaN

        # grace_time is represented in fixed-point s15.16 notation
        if self.grace_time is not None:
            _grace_time = int (self.grace_time *\
                               (1 << MLPConstants.FPREAL_SHIFT))
        else:
            _grace_time = MLPConstants.FPREAL_NaN

        return struct.pack("<4I",
                           _max_time,
                           _min_time,
                           _grace_time,
                           it_idx
                           )


class MLPEventValues ():
    """ a possibly named list of values associated with an event.
        The name corresponds to an INPUT group if the values are
        inputs, and to an OUTPUT group if the values are targets.
    """

    def __init__(self,
                 name = None
                 ):
        """
        """
        self.name = name

        # start with empty lists of values
        self.values = []
