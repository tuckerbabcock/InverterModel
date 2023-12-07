import numpy as np

import openmdao.api as om


class DCLinkCapacitor(om.ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input("I_phase_rms", units='A',
                       desc="The motor phase RMS current")
        self.add_input("modulation_index", units='unitless',
                       desc="Modulation index")
        self.add_input("power_factor", units='unitless',
                       desc="Power factor of the motor circuit accounting for external passive filters")
        self.add_input("f_sw", units='Hz',
                       desc="The inverter’s switching frequency")
        self.add_input("frequency", units='Hz',
                       desc="The inverter’s output electrical frequency")
        self.add_input("C", units='F',
                       desc="The capacitor's capacitance")
        self.add_input("dissipation_factor", units='unitless',
                       desc="The dissipation factor of the capacitor")

        self.add_output("V_ripple", units='V',
                        desc="Voltage ripple on the capacitor")
        self.add_output("P_loss", units='W',
                        desc="Losses in the capacitor due to the current ripple it experiences")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        I_phase_rms = inputs['I_phase_rms']
        modulation_index = inputs['modulation_index']
        power_factor = inputs['power_factor']
        f_sw = inputs['f_sw']
        frequency = inputs['frequency']
        C = inputs['C']
        dissipation_factor = inputs['dissipation_factor']

        I_in_rms = I_phase_rms * \
            np.sqrt(2 * np.sqrt(3) / np.pi *
                    modulation_index * (power_factor**2 + 0.25))
        I_in_avg = 0.75 * np.sqrt(2)*I_phase_rms * \
            modulation_index * power_factor

        I_cap_rms = np.sqrt(I_in_rms**2 - I_in_avg**2)

        outputs['V_ripple'] = I_cap_rms / (C * f_sw)

        R_cap_f = dissipation_factor / (2*np.pi*frequency*C)
        outputs['P_loss'] = I_cap_rms**2 * R_cap_f
