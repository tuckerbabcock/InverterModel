import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from invertermodel import Inverter


class TestInverter(unittest.TestCase):
    def test_inverter(self):
        prob = om.Problem()

        prob.model.add_subsystem("inverter",
                                 Inverter(),
                                 promotes=["*"])

        # prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        # prob.model.nonlinear_solver = om.NonlinearBlockGS()

        # prob.driver = om.ScipyOptimizeDriver()
        # prob.driver.options['optimizer'] = 'SLSQP'

        prob.driver = om.pyOptSparseDriver()

        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.options['debug_print'] = [
            'desvars', 'ln_cons', 'nl_cons', 'objs', 'totals']

        prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
        prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
        prob.driver.opt_settings['Verify level'] = -1
        prob.driver.opt_settings['Print file'] = "SNOPT.out"

        # Common design vars
        prob.model.add_design_var('I_phase_rms', lower=10)
        prob.model.add_design_var('r_wire', lower=0.001)

        # ### MOSFET
        # Design vars
        prob.model.add_design_var(
            'switching_frequency', lower=1.0, upper=1e6, ref=1e6)

        # ### Inductor
        # Design vars
        prob.model.add_design_var('ac_filter_inductor.n_turns', lower=1)
        prob.model.add_design_var(
            'ac_filter_inductor.R_core', lower=0.002, upper=0.1, ref0=0.002, ref=0.1)
        prob.model.add_design_var(
            'ac_filter_inductor.r_core', lower=0.001, ref0=0.001, ref=1.0)
        prob.model.add_design_var(
            'ac_filter_inductor.mu_r', lower=200, upper=1200)

        # Constraints
        prob.model.add_constraint(
            'ac_filter_inductor.radius_difference', lower=0.0001, linear=True)
        prob.model.add_constraint(
            'ac_filter_inductor.fill_factor', upper=0.5, ref=0.5)
        prob.model.add_constraint(
            'ac_filter_inductor.max_flux_density', upper=1.5)

        # ### Capacitor
        # Design vars
        prob.model.add_design_var('dc_link_cap.C', lower=0.0, ref=1e-4)

        prob.model.add_design_var('modulation_index_slack', upper=1.0)
        prob.model.add_constraint('modulation_index_balance', equals=0.0)
        prob.model.add_constraint('modulation_index', upper=1.0)

        prob.model.add_constraint('I_ripple', upper=0.05)
        prob.model.add_constraint('V_ripple', upper=0.01)

        prob.model.add_objective('efficiency', ref=-1)

        prob.setup()

        prob.set_val('load_inductance', 5.88007877e-5)
        prob.set_val('load_phase_back_emf', 946.36734443)
        prob.set_val('load_phase_resistance', 0.28172998)

        p0 = {
            'I_phase_rms': np.array([49.81200136]),
            'r_wire': np.array([0.00104543]),

            'electrical_frequency': 1727.18721061,

            'bus_voltage': 2000,
            'switching_frequency': 80000,

            'ac_filter_inductor.wire_density': 8960,
            'ac_filter_inductor.resistivity': 1.77e-8,
            'ac_filter_inductor.n_turns': 45.74874813,
            'ac_filter_inductor.R_core': 0.02,
            'ac_filter_inductor.r_core': 0.01,
            'ac_filter_inductor.mu_r': 1200,

            'dc_link_cap.C': 100e-6,
            'dc_link_cap.dissipation_factor': 140*1e-4,
            # 'dc_link_cap.dissipation_factor': 1000*1e-4,
            'dc_link_cap.specific_capacitance': 0.0006372145185838208,

            # https://assets.wolfspeed.com/uploads/2020/12/C2M0025120D.pdf
            'mosfet.R_ds_on': 0.025,
            'mosfet.Q_rr': 487e-9
        }

        for key, value in p0.items():
            prob[key][:] = value

        # prob.run_model()
        prob.run_driver()

        prob.model.list_inputs(units=True, prom_name=True)
        prob.model.list_outputs(residuals=True, units=True, prom_name=True)


if __name__ == "__main__":
    unittest.main()
