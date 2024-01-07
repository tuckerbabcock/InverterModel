import numpy as np

import openmdao.api as om


class RippleCurrent(om.ExplicitComponent):
    def setup(self):
        self.add_input("modulation_index", units='unitless',
                       desc="Modulation index")
        self.add_input("L", units='H',
                       desc="Phase inductance")
        self.add_input("switching_frequency", units='Hz',
                       desc="The inverter’s switching frequency")
        self.add_input("bus_voltage", units='V',
                       desc="DC link voltage")

        self.add_output("I_ripple", units='A',
                        desc="Ripple current at the output of the inverter")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        modulation_index = inputs['modulation_index']
        L = inputs['L']
        switching_frequency = inputs['switching_frequency']
        bus_voltage = inputs['bus_voltage']

        outputs['I_ripple'] = 0.5 * bus_voltage * \
            modulation_index / (2 * np.sqrt(3) * L * switching_frequency)
