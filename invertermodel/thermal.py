import numpy as np

import openmdao.api as om


class MOSFETThermalNetwork(om.ImplicitComponent):
    def setup(self):
        self.add_input("P_loss", units='W',
                       desc="The conduction and switching losses for a single switch")

        self.add_input("resistance_junction_to_case", units='K/W',
                       desc="Thermal resistance between the MOSFET junction and case")

        self.add_input("resistance_case_to_sink", units='K/W',
                       desc="Thermal resistance between the MOSFET case and heatsink")

        self.add_input("resistance_sink_to_air", units='K/W',
                       desc="Thermal resistance between the heatsink and ambient air")

        self.add_input("temperature_ambient", units='K',
                       desc="Temperature of the ambient air at the heatsink")

        self.add_output("temperature_junction", units='K',
                        desc="Temperature at the MOSFET junction")
        self.add_output("temperature_case", units='K',
                        desc="Temperature at the MOSFET case")
        self.add_output("temperature_sink", units='K',
                        desc="Temperature at the heatsink")

        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):
        Q = inputs['P_loss']
        R_jc = inputs['resistance_junction_to_case']
        R_cs = inputs['resistance_case_to_sink']
        R_sa = inputs['resistance_sink_to_air']

        T_j = outputs['temperature_junction']
        T_c = outputs['temperature_case']
        T_s = outputs['temperature_sink']
        T_a = inputs['temperature_ambient']

        residuals['temperature_junction'] = Q - (T_j - T_c) / R_jc
        residuals['temperature_case'] = (T_c - T_j) / R_jc - (T_c - T_s) / R_cs
        residuals['temperature_sink'] = (T_s - T_c) / R_cs - (T_s - T_a) / R_sa


class DCLinkCapacitorThermalNetwork(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("heatsink", default=False, types=bool,
                             desc="Indicates if the DC Capacitor is connected to a heatsink")

    def setup(self):
        self.add_input("P_loss", units='W',
                       desc="Losses in a single capacitor due to the current ripple it experiences")

        self.add_input("resistance_hotspot_to_case", units='K/W',
                       desc="Thermal resistance between the capacitor hotspot and case")

        heatsink = self.options['heatsink']
        if heatsink:
            self.add_input("resistance_case_to_sink", units='K/W',
                           desc="Thermal resistance between the capacitor case and heatsink")

            self.add_input("resistance_sink_to_air", units='K/W',
                           desc="Thermal resistance between the heatsink and ambient air")
        else:
            self.add_input("resistance_case_to_air", units='K/W',
                           desc="Thermal resistance between the capacitor case and heatsink")

        self.add_input("temperature_ambient", units='K',
                       desc="Temperature of the ambient air at the heatsink")

        self.add_output("temperature_hotspot", units='K',
                        desc="Temperature at the capacitor junction")
        self.add_output("temperature_case", units='K',
                        desc="Temperature at the capacitor case")

        if heatsink:
            self.add_output("temperature_sink", units='K',
                            desc="Temperature at the heatsink")

        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):
        heatsink = self.options['heatsink']
        Q = inputs['P_loss']
        R_hc = inputs['resistance_hotspot_to_case']
        if heatsink:
            R_cs = inputs['resistance_case_to_sink']
            R_sa = inputs['resistance_sink_to_air']
        else:
            R_ca = inputs['resistance_case_to_air']

        T_h = outputs['temperature_hotspot']
        T_c = outputs['temperature_case']

        if heatsink:
            T_s = outputs['temperature_sink']
        T_a = inputs['temperature_ambient']

        residuals['temperature_hotspot'] = Q - (T_h - T_c) / R_hc

        if heatsink:
            residuals['temperature_case'] = (
                T_c - T_h) / R_hc - (T_c - T_s) / R_cs
            residuals['temperature_sink'] = (
                T_s - T_c) / R_cs - (T_s - T_a) / R_sa
        else:
            residuals['temperature_case'] = (
                T_c - T_h) / R_hc - (T_c - T_a) / R_ca


class ACFilterInductorThermalNetwork(om.ImplicitComponent):
    def setup(self):
        self.add_input("P_loss_core", units='W',
                       desc="The core losses in a single inductor")
        self.add_input("P_loss_copper", units='W',
                       desc="The copper losses in a single inductor")

        self.add_input("resistance_core_to_windings", units='K/W',
                       desc="Thermal resistance between the inductor core and windings")

        self.add_input("resistance_windings_to_sink", units='K/W',
                       desc="Thermal resistance between the inductor windings and heatsink")

        self.add_input("resistance_sink_to_air", units='K/W',
                       desc="Thermal resistance between the heatsink and ambient air")

        self.add_input("temperature_ambient", units='K',
                       desc="Temperature of the ambient air at the heatsink")

        self.add_output("temperature_core", units='K',
                        desc="Temperature at the inductor core")
        self.add_output("temperature_windings", units='K',
                        desc="Temperature at the inductor windings")
        self.add_output("temperature_sink", units='K',
                        desc="Temperature at the heatsink")

        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):
        Q_core = inputs['P_loss_core']
        Q_copper = inputs['P_loss_copper']
        R_cw = inputs['resistance_core_to_windings']
        R_ws = inputs['resistance_windings_to_sink']
        R_sa = inputs['resistance_sink_to_air']

        T_c = outputs['temperature_core']
        T_w = outputs['temperature_windings']
        T_s = outputs['temperature_sink']
        T_a = inputs['temperature_ambient']

        residuals['temperature_core'] = Q_core - (T_c - T_w) / R_cw
        residuals['temperature_windings'] = Q_copper - \
            (T_c - T_w) / R_cw - (T_w - T_s) / R_ws
        residuals['temperature_sink'] = - \
            (T_w - T_s) / R_ws - (T_s - T_a) / R_sa
