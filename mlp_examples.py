class MLPExampleSet ():
    """ an MLP Lens-style example set
    """

    def __init__(self,
                 max_time    = None,
                 min_time    = None,
                 grace_time  = None,
                 def_input   = None,
                 def_target  = None,
                ):
        """
        """
        self.max_time   = max_time
        self.min_time   = min_time
        self.grace_time = grace_time
        self.def_input  = def_input
        self.def_target = def_target

        # start with an empty list of examples
        self.examples   = []

        print "creating example set"


class MLPExample ():
    """ an MLP Lens-style example
    """

    def __init__(self,
                 ex_id,
                 name = None
                 ):
        """
        """
        self.id   = ex_id
        self.name = name

        # start with an empty list of events
        self.events     = []

        print "creating example {}".format (ex_id)


class MLPEvent ():
    """ an MLP Lens-style event
    """

    def __init__(self,
                 ev_id,
                 max_time    = None,
                 min_time    = None,
                 grace_time  = None,
                 def_input   = None,
                 def_target  = None,
                 ):
        """
        """
        self.id         = ev_id
        self.max_time   = max_time
        self.min_time   = min_time
        self.grace_time = grace_time
        self.def_input  = def_input
        self.def_target = def_target

        print "creating event {}".format (ev_id)
