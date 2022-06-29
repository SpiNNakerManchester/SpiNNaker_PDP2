# Copyright (c) 2017-2019 The University of Manchester
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

from spinnaker_testbase import RootScriptBuilder


class ScriptBuilder(RootScriptBuilder):
    """
    This file will recreate the test_scripts.py file
    """

    def build_pdp2_scripts(self):
        #  see https:/github.com/SpiNNakerManchester/SpiNNaker_PDP2/issues/60
        skip_exceptions = {}
        skip_exceptions["simple_past_tense.py"] = [
            "from spinnman.exceptions import SpinnmanTimeoutException",
            "from spinnman.exceptions import SpiNNManCoresNotInStateException"]

        # create_test_scripts supports test that are too long or exceptions
        self.create_test_scripts(["examples"],
                                 skip_exceptions=skip_exceptions)


if __name__ == '__main__':
    builder = ScriptBuilder()
    builder.build_pdp2_scripts()
