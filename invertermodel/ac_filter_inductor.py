import numpy as np

import openmdao.api as om

from .inductor_core_materials import FE4491


class ACFilterInductor(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('core_material', default=FE4491,
                             desc='Dataclass that defines inductor core materials')

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
        self.add_input("mu_r", units='H/m',
                       desc="The relative permeability of the inductor core material")
        # self.add_input("core_density", units='kg/m**3',
        #                desc="The density of the inductor core material")

        self.add_discrete_input(
            "n_phases", val=3, desc="The number of inverter phases")

        self.add_output("max_flux_density", units='T',
                        desc="The maximum flux density in the inductor core")
        self.add_output("radius_difference", units='m',
                        desc='The difference between the toroid\'s major and minor radii, used to ensure a valid shape')
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
        core_material = self.options['core_material']

        I_phase_rms = inputs['I_phase_rms']
        electrical_frequency = inputs['electrical_frequency']
        resistivity = inputs['resistivity']
        wire_density = inputs['wire_density']

        n_turns = inputs['n_turns']
        r_wire = inputs['r_wire']

        R_core = inputs['R_core']
        r_core = inputs['r_core']
        mu_r = inputs['mu_r']
        # core_density = inputs['core_density']

        n_phases = discrete_inputs['n_phases']

        wire_area = np.pi * r_wire**2
        copper_area = n_turns * wire_area
        available_area = np.pi * (R_core-r_core)**2
        print(
            f"available_area: {available_area} {R_core-r_core}")
        outputs['radius_difference'] = R_core-r_core

        outputs['fill_factor'] = copper_area / available_area

        core_area = np.pi * r_core**2
        l_path = 2*np.pi*R_core
        mu = mu_r * 4*np.pi*1e-7
        outputs['inductance'] = mu * core_area * n_turns / l_path

        wire_volume = copper_area * l_path
        wire_mass = wire_volume * wire_density

        core_area = np.pi * r_core**2
        core_volume = core_area * 2 * np.pi * R_core

        core_density = core_material.density
        core_mass = core_volume * core_density
        outputs['mass'] = n_phases * wire_mass + core_mass

        B = mu * np.sqrt(2) * \
            I_phase_rms * n_turns / l_path
        # I_phase_rms * l_path / (n_turns * core_area)

        outputs['max_flux_density'] = B

        freq_scaler = 1e-3 if core_material.f_units == 'kHz' else 1.0
        steinmetz_params = core_material.steinmetz_params
        steinmetz_loss = steinmetz_params[0] * \
            (electrical_frequency *
             freq_scaler)**steinmetz_params[1] * B**steinmetz_params[2]
        outputs['P_loss_core'] = n_phases * steinmetz_loss * core_mass

        turn_length = 2*np.pi*r_core
        outputs['P_loss_copper'] = n_phases * n_turns * resistivity * \
            turn_length * I_phase_rms**2 / wire_area

        outputs['P_loss'] = outputs['P_loss_core'] + outputs['P_loss_copper']
