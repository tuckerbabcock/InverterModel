import numpy as np

import openmdao.api as om


class DCLinkCapacitor(om.ExplicitComponent):
    """
    Class that represents the combined effects of all of the DC link capacitors.
    It is assumed that multiple capacitors are connected together in parallel, with
    the specific number determined based on the input capacitance.
    """

    def initialize(self):
        pass

    def setup(self):
        self.add_input("I_phase_rms", units='A',
                       desc="The motor phase RMS current")
        self.add_input("modulation_index", units='unitless',
                       desc="Modulation index")
        self.add_input("power_factor", units='unitless',
                       desc="Power factor of the motor circuit accounting for external passive filters")
        self.add_input("switching_frequency", units='Hz',
                       desc="The inverter’s switching frequency")
        self.add_input("C", units='F',
                       desc="The DC link's total capacitance")
        self.add_input("dissipation_factor", units='unitless',
                       desc="The dissipation factor of the capacitor")
        self.add_input("specific_capacitance", units='F/kg',
                       desc="The specific capacitance of a single capacitor")

        self.add_output("V_ripple", units='V',
                        desc="Voltage ripple on the capacitor")
        self.add_output("P_loss", units='W',
                        desc="Losses in the capacitor due to the current ripple it experiences")
        self.add_output("mass", units='kg',
                        desc="The mass of all the DC link capacitors")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        I_phase_rms = inputs['I_phase_rms']
        modulation_index = inputs['modulation_index']
        power_factor = inputs['power_factor']
        switching_frequency = inputs['switching_frequency']
        C = inputs['C']
        dissipation_factor = inputs['dissipation_factor']
        specific_cap = inputs['specific_capacitance']

        I_in_rms = I_phase_rms * \
            np.sqrt(2 * np.sqrt(3) / np.pi *
                    modulation_index * (power_factor**2 + 0.25))
        I_in_avg = 0.75 * np.sqrt(2)*I_phase_rms * \
            modulation_index * power_factor

        if I_in_avg > I_in_rms:
            raise om.AnalysisError(
                f'Modulation index ({modulation_index}) too high! Insufficient bus voltage for given load')

        I_cap_rms = np.sqrt(I_in_rms**2 - I_in_avg**2)
        # print(
        #     f"I_in_rms: {I_in_rms}, I_in_avg: {I_in_avg}, I_cap_rms: {I_cap_rms}")

        outputs['V_ripple'] = I_cap_rms / (C * switching_frequency)

        R_cap_f = dissipation_factor / (2*np.pi*switching_frequency*C)
        outputs['P_loss'] = I_cap_rms**2 * R_cap_f

        outputs['mass'] = C / specific_cap
