# Copyright (c) 2015-2021 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
