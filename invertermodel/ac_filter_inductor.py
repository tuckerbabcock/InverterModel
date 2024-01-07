import numpy as np

import openmdao.api as om


class ACFilterInductor(om.ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input("I_phase_rms", units='A',
                       desc="The motor phase RMS current")
        self.add_input("electrical_frequency", units='Hz',
                       desc="The inverter’s output electrical frequency")
        self.add_input("resistivity", units='ohm*m',
                       desc="Resistivity of the conductor wire")
        self.add_input("wire_density", units='kg/m**3',
                       desc="The density of the conductor wire")

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
        self.add_input("core_density", units='kg/m**3',
                       desc="The density of the inductor core material")

        self.add_discrete_input(
            "n_phases", val=3, desc="The number of inverter phases")

        self.add_output("max_flux_density", units='T',
                        desc="The maximum flux density in the inductor core")
        self.add_output("fill_factor", units='unitless',
                        desc="The inductor winding fill factor")
        self.add_output("inductance", units='H',
                        desc="The inductance value of the toroidal inductor")
        self.add_output("mass", units='kg',
                        desc="The toroidal inductor's total mass")
        self.add_output("P_loss_core", units='W',
                        desc="Losses in the inductor due to core loss effects")
        self.add_output("P_loss_copper", units='W',
                        desc="Losses in the inductor due to resistive effects")
        self.add_output("P_loss", units='W',
                        desc="Losses in the inductor due to resistive and core loss effects")

        self.declare_partials('*', '*', method='cs')

        # self.declare_partials(
        #     'fill_factor', ['n_turns', 'r_wire', 'R_core', 'r_core'], method='cs')
        # self.declare_partials(
        #     'inductance', ['n_turns', 'R_core', 'r_core', 'mu'], method='cs')
        # self.declare_partials(
        #     'P_loss', ['I_phase_rms', 'resistivity', 'n_turns', 'r_wire', 'r_core'], method='cs')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        I_phase_rms = inputs['I_phase_rms']
        electrical_frequency = inputs['electrical_frequency']
        resistivity = inputs['resistivity']
        wire_density = inputs['wire_density']

        n_turns = inputs['n_turns']
        r_wire = inputs['r_wire']

        R_core = inputs['R_core']
        r_core = inputs['r_core']
        mu = inputs['mu']
        core_density = inputs['core_density']

        n_phases = discrete_inputs['n_phases']

        wire_area = np.pi * r_wire**2
        copper_area = n_turns * wire_area
        available_area = np.pi * (R_core**2 - R_core*r_core + r_core**2)

        outputs['fill_factor'] = copper_area / available_area

        core_area = np.pi * r_core**2
        l_path = 2*np.pi*R_core
        outputs['inductance'] = mu * core_area * n_turns / l_path

        wire_volume = copper_area * l_path
        wire_mass = wire_volume * wire_density

        core_area = np.pi * r_core**2
        core_volume = core_area * 2 * np.pi * R_core
        core_mass = core_volume * core_density
        outputs['mass'] = n_phases * wire_mass + core_mass

        B = np.sqrt(
            2) * I_phase_rms * l_path / (n_turns * core_area)
        outputs['max_flux_density'] = B

        steinmetz_loss = 0.3004 * \
            (electrical_frequency / 1000)**1.602 * B**2.085
        outputs['P_loss_core'] = n_phases * steinmetz_loss * core_mass

        turn_length = 2*np.pi*r_core
        outputs['P_loss_copper'] = n_phases * n_turns * resistivity * \
            turn_length * I_phase_rms**2 / wire_area

        outputs['P_loss'] = outputs['P_loss_core'] + outputs['P_loss_copper']
