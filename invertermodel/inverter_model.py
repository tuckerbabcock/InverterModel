import numpy as np

import openmdao.api as om

from .ac_filter_inductor import ACFilterInductor
from .dc_link_cap import DCLinkCapacitor
from .mosfet_loss import MOSFETLoss
from .ripple_current import RippleCurrent


class Inverter(om.Group):
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

        # ac_filter = om.Group()
        self.add_subsystem('ac_filter_inductor',
                           ACFilterInductor(),
                           promotes_inputs=['I_phase_rms',
                                            'r_wire',
                                            'n_phases'],
                           promotes_outputs=['fill_factor'])

        self.add_subsystem("combined_inductance",
                           om.ExecComp(
                               "L = load_inductance + filter_inductance",
                               L={"units": 'H'},
                               load_inductance={'units': 'H'},
                               filter_inductance={'units': 'H'}),
                           promotes_inputs=['load_inductance'],
                           promotes_outputs=['*'])
        self.connect('ac_filter_inductor.inductance',
                     'combined_inductance.filter_inductance')

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
                           promotes=['*'])

        self.add_subsystem("ripple_current",
                           RippleCurrent(),
                           promotes_inputs=['modulation_index',
                                            'L',
                                            'switching_frequency',
                                            'bus_voltage'],
                           promotes_outputs=['I_ripple'])

        self.add_subsystem("dc_link_cap",
                           DCLinkCapacitor(),
                           promotes_inputs=['I_phase_rms',
                                            'modulation_index',
                                            'power_factor',
                                            'switching_frequency',
                                            'electrical_frequency',
                                            # 'C',
                                            # 'dissipation_factor'
                                            ],
                           promotes_outputs=['V_ripple'])

        self.add_subsystem("total_loss",
                           om.ExecComp("total_loss = mosfet_loss + inductor_loss + capacitor_loss",
                                       total_loss={'units': 'W'},
                                       mosfet_loss={'units': 'W'},
                                       inductor_loss={'units': 'W'},
                                       capacitor_loss={'units': 'W'}),
                           promotes_outputs=['total_loss'])
        self.connect('mosfet.P_loss', 'total_loss.mosfet_loss')
        self.connect('ac_filter_inductor.P_loss', 'total_loss.inductor_loss')
        self.connect('dc_link_cap.P_loss', 'total_loss.capacitor_loss')
