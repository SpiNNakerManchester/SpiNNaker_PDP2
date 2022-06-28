# Copyright (c) 2019-2021 The University of Manchester
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

from spinnaker_testbase import ScriptChecker


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """
# flake8: noqa

    def test_examples_simple_past_tense_simple_past_tense(self):
        from spinnman.exceptions import SpinnmanTimeoutException
        from spinnman.exceptions import SpiNNManCoresNotInStateException
        self.check_script("examples/simple_past_tense/simple_past_tense.py", skip_exceptions=[SpinnmanTimeoutException,SpiNNManCoresNotInStateException])

    def test_examples_rogers_basic_rogers_basic(self):
        self.check_script("examples/rogers-basic/rogers-basic.py")

    def test_examples_rand10x40_rand10x40(self):
        self.check_script("examples/rand10x40/rand10x40.py")

    def test_examples_visSemPhon_visSemPhon(self):
        self.check_script("examples/visSemPhon/visSemPhon.py")
