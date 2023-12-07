import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from invertermodel import Inverter


class TestInverter(unittest.TestCase):
    def test_inverter(self):
        prob = om.Problem()

        prob.model.add_subsystem("inverter",
                                 Inverter(),
                                 promotes=["*"])

        prob.setup()
        prob.run_model()

        prob.model.list_inputs()
        prob.model.list_outputs()


if __name__ == "__main__":
    unittest.main()
