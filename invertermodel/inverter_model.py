import numpy as np

import openmdao.api as om

from .ac_filter_inductor import ACFilterInductor
from .dc_link_cap import DCLinkCapacitor
from .mosfet_loss import MOSFETLoss
from .ripple_current import RippleCurrent


class Inverter(om.Group):
    def initialize(self):
        self.options.declare("use_filter_inductor", default=True)

    def setup(self):
        # https://assets.wolfspeed.com/uploads/2020/12/C2M0025120D.pdf
        E_on_test = 2.18*1e-3
        E_off_test = 0.68*1e-3
        I_test = 63
        V_test = 1200
        self.add_subsystem("mosfet",
                           MOSFETLoss(E_on_test=E_on_test,
                                      E_off_test=E_off_test,
                                      I_test=I_test,
                                      V_test=V_test),
                           promotes_inputs=['I_phase_rms',
                                            'switching_frequency',
                                            'bus_voltage',
                                            'n_phases'])

        use_filter_inductor = self.options['use_filter_inductor']
        if use_filter_inductor:
            self.add_subsystem('ac_filter_inductor',
                               ACFilterInductor(),
                               promotes_inputs=['I_phase_rms',
                                                'r_wire',
                                                'n_phases',
                                                'electrical_frequency'])

            self.connect('ac_filter_inductor.inductance',
                         'combined_inductance.filter_inductance')
            self.connect('ac_filter_inductor.mass', 'mass.inductor_mass')
            self.connect('ac_filter_inductor.P_loss',
                         'total_loss.inductor_loss')

        self.add_subsystem("combined_inductance",
                           om.ExecComp(
                               "L = load_inductance + filter_inductance",
                               L={"units": 'H'},
                               load_inductance={'units': 'H'},
                               filter_inductance={'units': 'H', 'val': 0.0}),
                           promotes_inputs=['load_inductance'],
                           promotes_outputs=['*'])

        self.add_subsystem("phase_voltage",
                           om.ExecComp(
                               "phase_voltage = ((load_phase_back_emf + load_phase_resistance * (2**0.5)*I_phase_rms)**2 + (2*pi*L*electrical_frequency*(2**0.5)*I_phase_rms)**2)**0.5",
                               phase_voltage={'units': 'V'},
                               load_phase_back_emf={'units': 'V'},
                               load_phase_resistance={'units': 'ohm'},
                               I_phase_rms={'units': 'A'},
                               L={'units': 'H'},
                               electrical_frequency={'units': 'Hz'}),
                           promotes=['*'])

        # bal = om.BalanceComp()
        # bal.add_balance("phase_voltage_slack", eq_units='V',
        #                 lhs_name="phase_voltage", normalize=False)
        # self.add_subsystem("phase_voltage_slack",
        #                    bal,
        #                    promotes_inputs=['phase_voltage'],
        #                    promotes_outputs=['phase_voltage_slack'])

        # self.add_subsystem("phase_voltage_balance",
        #                    om.ExecComp("phase_voltage_balance = phase_voltage - phase_voltage_slack",
        #                                phase_voltage_balance={'units': 'V'},
        #                                phase_voltage={'units': 'V'},
        #                                phase_voltage_slack={'units': 'V'}),
        #                    promotes=['*'])

        self.add_subsystem("power_factor",
                           om.ExecComp(
                               "power_factor = load_phase_back_emf / phase_voltage",
                               power_factor={'units': 'unitless'},
                               load_phase_back_emf={'units': 'V'},
                               phase_voltage={'units': 'V'}),
                           promotes=['*'])

        self.add_subsystem("modulation_index",
                           om.ExecComp("modulation_index = 2 * phase_voltage / bus_voltage",
                                       modulation_index={'units': 'unitless'},
                                       phase_voltage={'units': 'V'},
                                       bus_voltage={'units': 'V'}),
                           promotes_inputs=['bus_voltage',
                                            'phase_voltage'],
                           promotes_outputs=['modulation_index'])

        self.add_subsystem("modulation_index_residual",
                           om.ExecComp("modulation_index_residual = modulation_index - modulation_index_slack",
                                       modulation_index_residual={
                                           'units': 'unitless'},
                                       modulation_index={'units': 'unitless'},
                                       modulation_index_slack={'units': 'unitless'}),
                           promotes=['*'])

        self.add_subsystem("ripple_current",
                           RippleCurrent(),
                           promotes_inputs=[
                               ('modulation_index',
                                   'modulation_index_slack'),
                               #    'modulation_index',
                               'L',
                               'switching_frequency',
                               'bus_voltage'])

        self.add_subsystem("dc_link_cap",
                           DCLinkCapacitor(),
                           promotes_inputs=['I_phase_rms',
                                            # 'modulation_index',
                                            ('modulation_index',
                                             'modulation_index_slack'),
                                            'power_factor',
                                            'switching_frequency',
                                            # 'C',
                                            # 'dissipation_factor'
                                            ])

        self.add_subsystem("ripple",
                           om.ExecComp([
                               "I_ripple = current_ripple / I_phase_rms",
                               "V_ripple = voltage_ripple / bus_voltage"
                           ],
                               current_ripple={'units': 'A'},
                               I_phase_rms={'units': 'A'},
                               I_ripple={'units': 'unitless'},
                               voltage_ripple={'units': 'V'},
                               bus_voltage={'units': 'V'},
                               V_ripple={'units': 'unitless'}),
                           promotes_inputs=['I_phase_rms', 'bus_voltage'],
                           promotes_outputs=['I_ripple', 'V_ripple'])
        self.connect('ripple_current.I_ripple', 'ripple.current_ripple')
        self.connect('dc_link_cap.V_ripple', 'ripple.voltage_ripple')

        self.add_subsystem("total_loss",
                           om.ExecComp("total_loss = mosfet_loss + inductor_loss + capacitor_loss",
                                       total_loss={'units': 'W'},
                                       mosfet_loss={'units': 'W'},
                                       inductor_loss={
                                           'units': 'W', 'val': 0.0},
                                       capacitor_loss={'units': 'W'}),
                           promotes_outputs=['total_loss'])
        self.connect('mosfet.P_loss', 'total_loss.mosfet_loss')
        self.connect('dc_link_cap.P_loss', 'total_loss.capacitor_loss')

        self.add_subsystem("power_out",
                           om.ExecComp("power_out = I_phase_rms * phase_voltage",
                                       power_out={'units': 'W'},
                                       I_phase_rms={'units': 'A'},
                                       phase_voltage={'units': 'V'}),
                           promotes=['*'])

        self.add_subsystem("efficiency",
                           om.ExecComp("efficiency = power_out / (power_out + total_loss)",
                                       efficiency={'units': 'unitless'},
                                       power_out={'units': 'W'},
                                       total_loss={'units': 'W'}),
                           promotes=['*'])

        self.add_subsystem('mass',
                           om.ExecComp('mass = inductor_mass + cap_mass',
                                       mass={'units': 'kg'},
                                       inductor_mass={
                                           'units': 'kg', 'val': 0.0},
                                       cap_mass={'units': 'kg'}),
                           promotes_outputs=['mass'])
        self.connect('dc_link_cap.mass', 'mass.cap_mass')

    # def configure(self):
    #     use_filer_inductor = self.options['use_filter_inductor']
    #     if use_filer_inductor:
    #         self.set_input_defaults('ac_filter_inductor.inductance', 1.0)
    #         self.set_input_defaults('ac_filter_inductor.P_loss', 1.0)
    #         self.set_input_defaults('ac_filter_inductor.mass', 1.0)
