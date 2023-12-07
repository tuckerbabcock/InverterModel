import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from invertermodel.dc_link_cap import DCLinkCapacitor


class TestDCLinkCapacitor(unittest.TestCase):
    def test_dc_link_cap_outputs(self):
        prob = om.Problem()

        prob.model.add_subsystem("dc_link_cap",
                                 DCLinkCapacitor(),
                                 promotes=["*"])

        prob.setup()
        prob.run_model()

        P_loss = prob.get_val("P_loss")
        self.assertAlmostEqual(P_loss[0], 0.12984214195024096)

    def test_dc_link_cap_partials(self):
        prob = om.Problem()

        prob.model.add_subsystem("dc_link_cap",
                                 DCLinkCapacitor(),
                                 promotes=["*"])

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)


if __name__ == "__main__":
    unittest.main()
