import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from invertermodel.mosfet_loss import MOSFETLoss


class TestMOSFETLoss(unittest.TestCase):
    def test_mosfet_loss(self):
        prob = om.Problem()

        E_on_test = 1.0
        E_off_test = 1.0
        I_test = 1.0
        V_test = 1.0

        prob.model.add_subsystem("mosfet",
                                 MOSFETLoss(E_on_test=E_on_test,
                                            E_off_test=E_off_test,
                                            I_test=I_test,
                                            V_test=V_test),
                                 promotes=["*"])

        prob.setup()
        prob.run_model()

        P_loss = prob.get_val("P_loss")
        self.assertAlmostEqual(P_loss[0], 9.901897896942637)

    def test_mosfet_loss_partials(self):
        prob = om.Problem()

        E_on_test = 1.0
        E_off_test = 1.0
        I_test = 1.0
        V_test = 1.0

        prob.model.add_subsystem("mosfet",
                                 MOSFETLoss(E_on_test=E_on_test,
                                            E_off_test=E_off_test,
                                            I_test=I_test,
                                            V_test=V_test),
                                 promotes=["*"])

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)


if __name__ == "__main__":
    unittest.main()
