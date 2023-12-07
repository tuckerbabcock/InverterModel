import numpy as np

import openmdao.api as om

class MOSFETLoss(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("E_on_test", desc="The turn-on energy loss given in the device datasheet for a specific bus voltage and load current")
        self.options.declare("E_off_test", desc="The turn-off energy loss given in the device datasheet for a specific bus voltage and load current")
        self.options.declare("I_test", desc="Test current given in the device datasheet for a specific bus voltage and load current")
        self.options.declare("V_test", desc="Test voltage given in the device datasheet for a specific bus voltage and load current")

    def setup(self):
        self.add_input("I_phase_rms", units='A', desc="The motor phase RMS current")
        self.add_input("R_ds_on", units='ohm', desc="Drain-source on-state resistance")
        self.add_input("f_sw", units='Hz', desc="The inverter’s switching frequency")
        self.add_input("V_bus", units='V', desc="DC link voltage")
        self.add_input("Q_rr", units='C', desc="Reverse recovery charge")

        self.add_discrete_input("n_phases", val=3, desc="The number of inverter phases")
        self.add_discrete_input("switches_per_phase", val=2, desc="The number of MOSFETs per phase")

        self.add_output("P_loss", units='W', desc="Sum of all of the conduction and switching losses")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        E_on_test = self.options['E_on_test']
        E_off_test = self.options['E_off_test']
        I_test = self.options['I_test']
        V_test = self.options['V_test']

        I_phase_rms = inputs['I_phase_rms']
        R_ds_on = inputs['R_ds_on']
        f_sw = inputs['f_sw']
        V_bus = inputs['V_bus']
        Q_rr = inputs['Q_rr']

        n_phases = discrete_inputs['n_phases']
        switches_per_phase = discrete_inputs['switches_per_phase']

        P_cond = 0.5 * I_phase_rms**2 * R_ds_on
        
        P_switch = f_sw * (np.sqrt(2) / np.pi * I_phase_rms) * V_bus
        P_on = P_switch * E_on_test / (I_test * V_test)
        P_off = P_switch * E_off_test / (I_test * V_test)

        P_rr = 0.25 * Q_rr * V_bus * f_sw

        outputs['P_loss'] = n_phases * switches_per_phase * (P_cond + P_on + P_off + P_rr)
