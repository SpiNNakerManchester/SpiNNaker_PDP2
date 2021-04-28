class MLPLink ():
    """ an MLP link
    """

    def __init__(self,
                 pre_link_group  = None,
                 post_link_group = None,
                 label           = None,
                 VERBOSE         = False
                 ):
        """
        """
        self.pre_link_group  = pre_link_group
        self.post_link_group = post_link_group
        self.label           = label

        if VERBOSE: print (f"creating link {self.label}")

        # update list of incoming links in the post_link_group
        self.post_link_group.links_from.append (self.pre_link_group)
