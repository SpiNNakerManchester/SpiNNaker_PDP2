class MLPLink ():
    """ an MLP link
    """

    def __init__(self,
                 pre_link_group  = None,
                 post_link_group = None,
                 label = None
                 ):
        """
        """
        self._pre_link_group  = pre_link_group
        self._post_link_group = post_link_group
        self._label           = label

        print "creating link {}".format (self._label)

    @property
    def label (self):
        return self._label

    @property
    def pre_link_group (self):
        return self._pre_link_group

    @property
    def post_link_group (self):
        return self._post_link_group
