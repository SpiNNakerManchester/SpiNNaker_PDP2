import os
import struct

from spinn_pdp2.mlp_types import MLPConstants


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
        self.num_examples = 0

        # track if examples have been loaded
        self.examples_loaded = False

        # track if examples have been compiled
        self.examples_compiled = False

        print ("creating example set")


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
            print (f"setting max_time to {max_time}")
            self.max_time = max_time

        if min_time is not None:
            print (f"setting min_time to {min_time}")
            self.min_time = min_time

        if grace_time is not None:
            print (f"setting grace_time to {grace_time}")
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
        # max_time is an MLP fixed-point fpreal
        if (self.max_time is None) or (self.max_time == float ('nan')):
            max_time = MLPConstants.FPREAL_NaN
        else:
            max_time = int (self.max_time *\
                            (1 << MLPConstants.FPREAL_SHIFT))

        # min_time is an MLP fixed-point fpreal
        if (self.min_time is None) or (self.min_time == float ('nan')):
            min_time = MLPConstants.FPREAL_NaN
        else:
            min_time = int (self.min_time *\
                            (1 << MLPConstants.FPREAL_SHIFT))

        # grace_time is an MLP fixed-point fpreal
        if (self.grace_time is None) or (self.grace_time == float ('nan')):
            grace_time = MLPConstants.FPREAL_NaN
        else:
            grace_time = int (self.grace_time *\
                            (1 << MLPConstants.FPREAL_SHIFT))

        return struct.pack("<4I",
                           len (self.examples),
                           max_time,
                           min_time,
                           grace_time
                           )


    def read_Lens_examples_file (self,
                                 examples_file
                                 ):
        """ reads a Lens-style examples file

        Lens online manual @ CMU:
            https://ni.cmu.edu/~plaut/Lens/Manual/

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
        else:
            self._examples_file = None
            print (f"error: cannot open examples file: {examples_file}")
            return

        print ("reading Lens-style examples file")

        ef = open (self._examples_file, "r")

        print ("processing example set header")

        # process example set header
        line = ef.readline ()
        while (';' not in line):
            if ('proc:' in line):
                print ("set procedure not supported")
            elif ('max:' in line):
                _, val = line.split (':')
                try:
                    self.max_time = float (val)
                except:
                    self.max_time = float ('nan')
                print (f"setting set max:{self.max_time}")
            elif ('min:' in line):
                _, val = line.split (':')
                try:
                    self.min_time = float (val)
                except:
                    self.min_time = float ('nan')
                print (f"setting set min:{self.min_time}")
            elif ('grace:' in line):
                _, val = line.split (':')
                try:
                    self.grace_time = float (val)
                except:
                    self.grace_time = float ('nan')
                print (f"setting set grace:{self.grace_time}")
            elif ('defI:' in line):
                _, val = line.split (':')
                try:
                    self.def_input = float (val)
                except:
                    self.def_input = float ('nan')
                print (f"setting set defI:{self.def_input}")
            elif ('actI:' in line):
                print ("set active input not supported")
            elif ('defT:' in line):
                _, val = line.split (':')
                try:
                    self.def_target = float (val)
                except:
                    self.def_target = float ('nan')
                print (f"setting set defT:{self.def_target}")
            elif ('actT:' in line):
                print ("set active target not supported")
            else:
                # ';' is optional
                break

            line = ef.readline ()

        # ';' is optional
        if (';' in line):
            line = ef.readline ()

        # process each example in the set
        ex_id = 0
        while (line != ""):
            # create new example, initially empty
            _ex = MLPExample (ex_id)

            print (f"processing example {ex_id}")

            # process the example header
            done = False
            while not done:
                if (line.strip () == ''):
                    print ("ignoring empty line")
                elif ('name:' in line):
                    _, _name = line.split (':')
                    _ex.name = _name.rstrip ()
                    print (f"setting example name:{_ex.name}")

                elif ('proc:' in line):
                    print ("example procedure not supported")

                elif ('freq:' in line):
                    _, freq = line.split (':')
                    _ex.freq = float (freq)
                    print (f"setting example freq:{_ex.freq}")

                else:
                    # try to get number of events
                    try:
                        num_ev = int (line)
                        done = True
                        print (f"setting example num_ev:{num_ev}")
                    except:
                        # if absent, number of events defaults to 1
                        num_ev = 1
                        print (f"setting (default) example num_ev:{num_ev}")
                        break

                # prepare to process next line
                line = ef.readline ()
                if (line == ""):
                    print ("error: unexpected end-of-file")
                    ef.close ()
                    return None

            # instantiate an event container for every event in example
            events = []
            for i in range (num_ev):
                # instantiate new empty event and add it to list
                _ev = MLPEvent (i)
                events.append (_ev)

            # process each event in the example
            ev_iid = 0
            ev_tid = 0
            while True:
                # TODO: need to complete event list processing!
                # read event list, if present
                if ("[" in line):
                    # compose the multi-line list
                    evl = line.strip ()
                    while ("]" not in line):
                        line = ef.readline ()
                        evl += " " + line.strip ()

                    ev_list, line = evl.split (']')

                    print (f"processing event list {ev_list}")

                    maxt = None
                    mint = None
                    grct = None
                    defi = None
                    deft = None

                    for s in (ev_list.strip ("[]")).split ():
                        if ('proc:' in s):
                            print ("event procedure not supported")

                        elif ('max:' in s):
                            _, val = s.split (':')
                            try:
                                maxt = float (val)
                            except:
                                maxt = float ('nan')
                            print (f"setting event max:{maxt}")

                        elif ('min:' in s):
                            _, val = s.split (':')
                            try:
                                mint = float (val)
                            except:
                                mint = float ('nan')
                            print (f"setting event min:{mint}")

                        elif ('grace:' in s):
                            _, val = s.split (':')
                            try:
                                grct = float (val)
                            except:
                                grct = float ('nan')
                            print (f"setting event grace:{grct}")

                        elif ('defI:' in s):
                            _, val = s.split (':')
                            try:
                                defi = float (val)
                            except:
                                defi = float ('nan')
                            print (f"setting event defI:{defi}")

                        elif ('actI:' in s):
                            print ("event active Input not supported")

                        elif ('defT:' in s):
                            _, val = s.split (':')
                            try:
                                deft = float (val)
                            except:
                                deft = float ('nan')
                            print (f"setting event defT:{deft}")

                        elif ('actT:' in s):
                            print ("event active Target not supported")

                        elif ('*' in s):
                            print ("multi-event lists not supported")

                        elif ('-' in s):
                            print ("multi-event lists not supported")

                        else:
                            ev_act = int (s)
                            ev_iid = ev_act
                            ev_tid = ev_act
                            print (f"event in event list: {ev_act}")

                    if maxt is not None:
                        events[ev_act].max_time = maxt

                    if mint is not None:
                        events[ev_act].min_time = mint

                    if grct is not None:
                        events[ev_act].grace_time = grct

                    if defi is not None:
                        events[ev_act].def_input = defi

                    if deft is not None:
                        events[ev_act].def_target = deft

                # get inputs and targets for every event
                # check line for inputs
                if ('I:' in line) or ('i:' in line):
                    print (f"reading event {ev_iid}/-")

                    # remove line identifier
                    _, _is = (line.rstrip (" ;\n")).split (":")

                    # check if multiple group inputs
                    if "(" in _is:
                        # split into group inputs
                        _gis = _is.split ("(")

                        # process each group in turn
                        for grpi in _gis[1:]:
                            # create list of values
                            _gi = grpi.split ()

                            # instantiate a new event value container
                            vl = MLPEventValues ()

                            # first element is group name
                            vl.name = _gi[0].rstrip (")")

                            # rest are input values
                            for v in _gi[1:]:
                                vl.values.append (float (v))

                            # store inputs in event
                            print (f"added inputs {vl.name}:{vl.values}")
                            events[ev_iid].inputs.append (vl)
                    else:
                        # instantiate a new event value container
                        vl = MLPEventValues ()

                        # no group name given
                        vl.name = None

                        # read input values
                        for v in _is.split ():
                            vl.values.append (float (v))

                        # store inputs in event
                        print (f"added inputs {vl.name}:{vl.values}")
                        events[ev_iid].inputs.append (vl)

                    # update event input index
                    ev_iid += 1

                # check line for targets
                elif ('T:' in line) or ('t:' in line):
                    print (f"reading event -/{ev_tid}")

                    # remove line identifier
                    _, _ts = (line.rstrip (" ;\n")).split (":")

                    # check if multiple group inputs
                    if "(" in _ts:
                        # split into group inputs
                        _gts = _ts.split ("(")

                        # process each group in turn
                        for grpt in _gts[1:]:
                            # create list of values
                            _gt = grpt.split ()

                            # instantiate a new event value container
                            vl = MLPEventValues ()

                            # first element is group name
                            vl.name = _gt[0].rstrip (")")

                            # rest are target values
                            for v in _gt[1:]:
                                vl.values.append (float (v))

                            # store targets in event
                            print (f"added targets {vl.name}:{vl.values}")
                            events[ev_tid].targets.append (vl)
                    else:
                        # instantiate a new event value container
                        vl = MLPEventValues ()

                        # no group name given
                        vl.name = None

                        # read target values
                        for v in _ts.split ():
                            vl.values.append (float (v))

                        # store targets in event
                        print (f"added targets {vl.name}:{vl.values}")
                        events[ev_tid].targets.append (vl)

                    # update event target index
                    ev_tid += 1

                # check line for both inputs and targets
                elif ('B:' in line) or ('b:' in line):
                    print (f"reading event {ev_iid}/{ev_tid}")

                    # remove line identifier
                    _, _is = (line.rstrip (" ;\n")).split (":")

                    # check if multiple group inputs
                    if "(" in _is:
                        # split into group inputs
                        gis = _is.split (":")

                        # process each group in turn
                        for grpi in gis[1:]:
                            # create list of values
                            gi = grpi.split ()

                            # instantiate a new event value container
                            vl = MLPEventValues ()

                            # first element is group name
                            vl.name = gi[0].rstrip (")")

                            # rest are input values
                            for v in gi[1:]:
                                vl.values.append (float (v))

                            # store inputs and targets in event
                            print (f"added inputs/targets {vl.name}:{vl.values}")
                            events[ev_iid].inputs.append (vl)
                            events[ev_tid].targets.append (vl)
                    else:
                        # instantiate a new event value container
                        vl = MLPEventValues ()

                        # no group name given
                        vl.name = None

                        # read input values
                        for v in _is.split ():
                            vl.values.append (float (v))

                        # store inputs and targets in event
                        print (f"added inputs/targets {vl.name}:{vl.values}")
                        events[ev_iid].inputs.append (vl)
                        events[ev_tid].targets.append (vl)

                    # update event input and event target indices
                    ev_iid += 1
                    ev_tid += 1

                # check if final event in example
                if (";" in line):
                    break

                # prepare to process new line
                line = ef.readline ()
                if (line == ""):
                    print ("error: unexpected end-of-file")
                    ef.close ()
                    return None

            # add events to example event list
            for ev in events:
                print (f"adding event {ev.id} to example {_ex.id}")
                _ex.events.append (ev)

            # add example to set example list
            self.examples.append (_ex)

            # prepare for next example
            ex_id += 1
            line = ef.readline ()

        # clean up
        ef.close ()

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
        print (f"compiling {self.label}")

        # create example set configuration
        self.set_config = self.config

        # create example and event configuration lists
        self.example_config = []
        self.event_config   = []

        ev_idx = 0
        it_idx = 0

        # create a configuration for every example
        for n, _ex in enumerate (self.examples):
            self.example_config.append (_ex.config (n, ev_idx))
            ev_idx += len (_ex.events)

            # process every event in the example
            for _ev in _ex.events:
                # create an event configuration
                self.event_config.append (_ev.config (it_idx))
                it_idx += 1

                # set default input value for this event
                if _ev.def_input is not None:
                    defi = _ev.def_input
                elif self.def_input is not None:
                    defi = self.def_input
                else:
                    defi = None

                # set default target value for this event
                if _ev.def_target is not None:
                    deft = _ev.def_target
                elif self.def_target is not None:
                    deft = self.def_target
                else:
                    deft = None

                # process event inputs
                grps_done = []
                for ei in _ev.inputs:
                    if ei.name is None:
                        if len (self._net.in_grps) == 1:
                            self._net.in_grps[0].inputs += ei.values
                            grps_done.append(self._net.in_grps[0])
                        else:
                            print ("error: not enough inputs in event")
                            return 0
                    else:
                        for g in self._net.in_grps:
                            if g.label == ei.name:
                                grps_done.append (g)
                                g.inputs += ei.values
                                break

                # add default inputs for not-listed groups
                for g in self._net.in_grps:
                    if g not in grps_done:
                        for _ in range (g.units):
                            g.inputs.append (defi)

                    print (f"{g.label}: {len (g.inputs)} inputs")

                # process event targets
                grps_done = []
                for et in _ev.targets:
                    if et.name is None:
                        if len (self._net.out_grps) == 1:
                            self._net.out_grps[0].targets += et.values
                            grps_done.append(self._net.out_grps[0])
                        else:
                            print ("error: not enough targets in event")
                            return 0
                    else:
                        for g in self._net.out_grps:
                            if g.label == et.name:
                                grps_done.append (g)
                                g.targets += et.values
                                break

                # add default targets for not-listed groups
                for g in self._net.out_grps:
                    if g not in grps_done:
                        for _ in range (g.units):
                            g.targets.append (deft)

                    print (f"{g.label}: {len (g.targets)} targets")

        self.num_examples = len (self.examples)

        # mark examples file as compiled
        self.examples_compiled = True

        return self.num_examples


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

        print (f"creating example {ex_id}")


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
        # freq is an MLP fixed-point fpreal
        if self.freq is not None:
            freq = int (self.freq * (1 << MLPConstants.FPREAL_SHIFT))
        else:
            freq = int (MLPConstants.DEF_EX_FREQ *\
                         (1 << MLPConstants.FPREAL_SHIFT))

        return struct.pack("<4I",
                           num,
                           len (self.events),
                           ev_idx,
                           freq
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

        print (f"creating event {ev_id}")


    def config (self, it_idx):
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
        # max_time is an MLP fixed-point fpreal
        if (self.max_time is None) or (self.max_time == float ('nan')):
            max_time = MLPConstants.FPREAL_NaN
        else:
            max_time = int (self.max_time *\
                            (1 << MLPConstants.FPREAL_SHIFT))

        # min_time is an MLP fixed-point fpreal
        if (self.min_time is None) or (self.min_time == float ('nan')):
            min_time = MLPConstants.FPREAL_NaN
        else:
            min_time = int (self.min_time *\
                            (1 << MLPConstants.FPREAL_SHIFT))

        # grace_time is an MLP fixed-point fpreal
        if (self.grace_time is None) or (self.grace_time == float ('nan')):
            grace_time = MLPConstants.FPREAL_NaN
        else:
            grace_time = int (self.grace_time *\
                            (1 << MLPConstants.FPREAL_SHIFT))

        return struct.pack("<4I",
                           max_time,
                           min_time,
                           grace_time,
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
