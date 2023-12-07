import numpy as np

import openmdao.api as om


class ACFilterInductor(om.ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input("I_phase_rms", units='A',
                       desc="The motor phase RMS current")
        self.add_input("resistivity", units='ohm*m',
                       desc="Resistivity of the conductor wire")

        self.add_input("n_turns", units='unitless',
                       desc="Number of wire turns wrapping the inductor core")
        self.add_input("r_wire", units='m',
                       desc="The radius of the wire that wraps the inductor core")

        self.add_input("R_core", units='m',
                       desc="The major radius of the toroidal inductor core")
        self.add_input("r_core", units='m',
                       desc="The minor radius of the toroidal inductor core")
        self.add_input("mu", units='H/m',
                       desc="The effective permeability of the inductor core material")

        self.add_discrete_input(
            "n_phases", val=3, desc="The number of inverter phases")

        self.add_output("fill_factor", units='unitless',
                        desc="The inductor winding fill factor")
        self.add_output("inductance", units='H',
                        desc="The inductance value of the toroidal inductor")
        self.add_output("P_loss", units='W',
                        desc="Losses in the inductor due to resistive and core loss effects")

        self.declare_partials(
            'fill_factor', ['n_turns', 'r_wire', 'R_core', 'r_core'], method='cs')
        self.declare_partials(
            'inductance', ['n_turns', 'R_core', 'r_core', 'mu'], method='cs')
        self.declare_partials(
            'P_loss', ['I_phase_rms', 'resistivity', 'n_turns', 'r_wire', 'r_core'], method='cs')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        I_phase_rms = inputs['I_phase_rms']
        resistivity = inputs['resistivity']

        n_turns = inputs['n_turns']
        r_wire = inputs['r_wire']

        R_core = inputs['R_core']
        r_core = inputs['r_core']
        mu = inputs['mu']

        n_phases = discrete_inputs['n_phases']

        wire_area = np.pi * r_wire**2
        copper_area = n_turns * wire_area
        available_area = np.pi * (R_core**2 - R_core*r_core + r_core**2)

        outputs['fill_factor'] = copper_area / available_area

        core_area = np.pi * r_core**2
        l_path = 2*np.pi*R_core
        outputs['inductance'] = mu * core_area * n_turns / l_path

        turn_length = 2*np.pi*r_core
        outputs['P_loss'] = n_phases * n_turns * resistivity * \
            turn_length * I_phase_rms**2 / wire_area
