# pinnacle/physics.py
"""
Electrochemical physics parameters, calculations, and PDE formulations for PINNACLE.
"""

import torch
from typing import Dict, Any, NamedTuple, Tuple
from dataclasses import dataclass
from .gradients import GradientComputer, GradientConfig


@dataclass
class PhysicsConstants:
    """Fundamental physical constants"""
    F: float = 96485.0  # Faraday constant [C/mol]
    R: float = 8.3145  # Gas constant [J/(mol·K)]
    T: float = 293.0  # Temperature [K]
    k_B: float = 1.3806e-23  # Boltzmann constant [J/K]
    eps0: float = 8.85e-12  # Vacuum permittivity [F/m]
    electron_charge: float = 1.6e-19  # Elementary charge [C]


@dataclass
class TransportProperties:
    """Transport properties for different species"""
    # Diffusion coefficients [m²/s]
    D_cv: float = 1.0e-21
    D_av: float = 1.0e-21
    D_h: float = 3.2823e-4

    # Mobility coefficients [m²/(V·s)]
    U_cv: float = -1.0562e-19
    U_av: float = 7.9212e-20
    U_h: float = 0.013

    # Species charges
    z_cv: float = -2.6667  # -8/3
    z_av: float = 2.0
    z_h: float = 1.0


@dataclass
class MaterialProperties:
    """Material and solution properties"""
    # Permittivities [F/m]
    epsilonf: float = 1.239e-10  # 14*eps0
    eps_film: float = 1.239e-10  # Same as epsilonf
    eps_Ddl: float = 1.77e-11  # 2*eps0
    eps_dl: float = 6.947e-10  # 78.5*eps0
    eps_sol: float = 6.947e-10  # Same as eps_dl

    # Semiconductor properties
    c_h0: float = 4.1683e-4  # Intrinsic hole concentration [mol/m³]
    c_e0: float = 9.5329e-28  # Intrinsic electron concentration [mol/m³]
    tau: float = 4.9817e-13  # Recombination time constant [s·mol/m³]
    Nc: float = 166.06  # Conduction band density [mol/m³]
    Nv: float = 1.6606e5  # Valence band density [mol/m³]
    mu_e0: float = 2.4033e-19  # Standard electron chemical potential [J]
    Ec0: float = 5.127e-19  # Conduction band edge [J]
    Ev0: float = 1.6022e-19  # Valence band edge [J]

    # Solution properties
    c_H: float = 0.01  # Proton concentration [mol/m³]
    pH: float = 5.0
    Omega: float = 1.4e-5  # Molar volume [m³/mol]


@dataclass
class ReactionKinetics:
    """Reaction rate constants and kinetic parameters"""
    # Standard rate constants
    k1_0: float = 4.5e-8  # [m/s]
    k2_0: float = 3.6e-6  # [mol/(m²·s)]
    k3_0: float = 4.5e-9  # [mol/(m²·s)]
    k4_0: float = 2.25e-7  # [m/s]
    k5_0: float = 7.65e-9  # [mol/(m²·s)]
    ktp_0: float = 4.5e-8  # [-]
    ko2_0: float = 0.005  # [m/s]

    # Charge transfer coefficients
    alpha_cv: float = 0.3
    alpha_av: float = 0.8
    beta_cv: float = 0.1
    beta_av: float = 0.8
    alpha_tp: float = 0.2
    a_par: float = 0.45  # For oxygen evolution

    # Derived parameters [1/V] - computed from alpha/beta and F/(R*T)
    a_cv: float = 23.764  # alpha_cv * 2 * F/(R*T)
    a_av: float = 84.493  # alpha_av * 8/3 * F/(R*T)
    b_cv: float = 7.9212  # beta_cv * 2 * F/(R*T)

    # Equilibrium potentials
    phi_O2_eq: float = 1.35  # [V]


@dataclass
class GeometryParameters:
    """Geometric parameters and dimensions"""
    d_Ddl: float = 2.0e-10  # Defect layer thickness [m]
    d_dl: float = 5.0e-10  # Double layer thickness [m]
    L_cell: float = 1.0e-6  # Cell length [m]

    # Applied potential range
    E_min: float = -1.0  # [V]
    E_max: float = 1.8  # [V]


@dataclass
class DomainParameters:
    """Domain and initial conditions"""
    time_scale: float = 3600.0  # [s]
    L_initial: float = 1e-9  # Initial film thickness [m]
    delta3: float = 1.0  # Chemical parameter


@dataclass
class CharacteristicScales:
    """Characteristic scales for non-dimensionalization"""
    lc: float = 1e-9  # Length scale [m]
    cc: float = 1e-5  # Concentration scale [mol/m³]

    # Computed scales (will be calculated in post_init)
    tc: float = None  # Time scale [s]
    phic: float = None  # Potential scale [V]
    chc: float = None  # Hole concentration scale [mol/m³]


class ElectrochemicalPhysics:
    """
        Main physics manager that handles all electrochemical calculations.

        This class consolidates all physics parameters and provides methods
        for PDE calculations, rate constant computations, and scaling.
        """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize physics parameters from configuration.

        Args:
            config: Configuration dictionary
            device: PyTorch device for computations
        """
        self.device = device
        self.config = config

        # Load all parameter groups from config
        self._load_parameters_from_config()

        # Set up characteristic scales
        self._setup_characteristic_scales()

        # Initialize gradient computer
        grad_config = GradientConfig(
            create_graph=True,
            retain_graph=True,
            validate_inputs=True
        )

        self.grad_computer = GradientComputer(grad_config, device)

    def _load_parameters_from_config(self):
        """Load all physics parameters from configuration"""
        pde_config = self.config['pde']

        # Load each parameter group
        self.constants = PhysicsConstants(
            F=pde_config['physics']['F'],
            R=pde_config['physics']['R'],
            T=pde_config['physics']['T'],
            k_B=pde_config['physics']['k_B'],
            eps0=pde_config['physics']['eps0'],
            electron_charge=1.6e-19  # Not in config, using default
        )

        self.transport = TransportProperties(
            D_cv=pde_config['physics']['D_cv'],
            D_av=pde_config['physics']['D_av'],
            D_h=pde_config['physics']['D_h'],
            U_cv=pde_config['physics']['U_cv'],
            U_av=pde_config['physics']['U_av'],
            U_h=pde_config['physics']['U_h'],
            z_cv=pde_config['physics']['z_cv'],
            z_av=pde_config['physics']['z_av'],
            z_h=pde_config['physics']['z_h']
        )

        self.materials = MaterialProperties(
            epsilonf=pde_config['physics']['epsilonf'],
            eps_film=pde_config['physics']['eps_film'],
            eps_Ddl=pde_config['physics']['eps_Ddl'],
            eps_dl=pde_config['physics']['eps_dl'],
            eps_sol=pde_config['physics']['eps_sol'],
            c_h0=pde_config['physics']['c_h0'],
            c_e0=pde_config['physics']['c_e0'],
            tau=pde_config['physics']['tau'],
            Nc=pde_config['physics']['Nc'],
            Nv=pde_config['physics']['Nv'],
            mu_e0=pde_config['physics']['mu_e0'],
            Ec0=pde_config['physics']['Ec0'],
            Ev0=pde_config['physics']['Ev0'],
            c_H=pde_config['physics']['c_H'],
            pH=pde_config['physics']['pH'],
            Omega=pde_config['physics']['Omega']
        )

        self.kinetics = ReactionKinetics(
            k1_0=pde_config['rates']['k1_0'],
            k2_0=pde_config['rates']['k2_0'],
            k3_0=pde_config['rates']['k3_0'],
            k4_0=pde_config['rates']['k4_0'],
            k5_0=pde_config['rates']['k5_0'],
            ktp_0=pde_config['rates']['ktp_0'],
            ko2_0=pde_config['rates']['ko2_0'],
            alpha_cv=pde_config['rates']['alpha_cv'],
            alpha_av=pde_config['rates']['alpha_av'],
            beta_cv=pde_config['rates']['beta_cv'],
            beta_av=pde_config['rates']['beta_av'],
            alpha_tp=pde_config['rates']['alpha_tp'],
            a_par=pde_config['rates']['a_par'],
            a_cv=pde_config['rates']['a_cv'],
            a_av=pde_config['rates']['a_av'],
            b_cv=pde_config['rates']['b_cv'],
            phi_O2_eq=pde_config['rates']['phi_O2_eq']
        )

        self.geometry = GeometryParameters(
            d_Ddl=pde_config['geometry']['d_Ddl'],
            d_dl=pde_config['geometry']['d_dl'],
            L_cell=pde_config['geometry']['L_cell'],
            E_min=pde_config['physics']['E_min'],
            E_max=pde_config['physics']['E_max']
        )

        self.domain = DomainParameters(
            time_scale=self.config['domain']['time']['time_scale'],
            L_initial=self.config['domain']['initial']['L_initial'],
            delta3=pde_config['chemistry']['delta3']
        )

        self.scales = CharacteristicScales(
            lc=pde_config['scales']['lc'],
            cc=pde_config['scales']['cc']
        )

    def _setup_characteristic_scales(self):
        """Setup characteristic scales for non-dimensionalization"""
        # Compute derived scales
        self.scales.tc = self.scales.lc ** 2 / self.transport.D_cv
        self.scales.phic = self.constants.R * self.constants.T / self.constants.F
        self.scales.chc = self.materials.c_h0

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of all physics parameters"""
        return {
            'constants': self.constants.__dict__,
            'transport': self.transport.__dict__,
            'materials': self.materials.__dict__,
            'kinetics': self.kinetics.__dict__,
            'geometry': self.geometry.__dict__,
            'domain': self.domain.__dict__,
            'scales': self.scales.__dict__
        }

    def to_tensor(self, value: float) -> torch.Tensor:
        """Convert scalar to tensor on correct device"""
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def compute_rate_constants(self, t: torch.Tensor, E: torch.Tensor, networks, single: bool = False):
        """
        Compute electrochemical rate constants using Butler-Volmer kinetics.

        **Butler-Volmer Rate Expressions:**

        .. math::
            \\hat{k}_{R1} = k_1^0 \\exp\\left(\\alpha_1 \\frac{3F\\hat{\\phi}_c}{RT}\\hat{\\phi}_{mf} \\right)

        .. math::
            \\hat{k}_{R2} = k_2^0 \\exp\\left(\\alpha_2 \\frac{2F\\hat{\\phi}_c}{RT}\\hat{\\phi}_{mf} \\right)

        .. math::
            \\hat{k}_{R3} = k_3^0 \\exp\\left(\\alpha_3 \\frac{(3-\\delta)F\\hat{\\phi}_c}{RT}\\hat{\\phi}_{fs} \\right)

        .. math::
            \\hat{k}_{R4} = k_4^0

        .. math::
            \\hat{k}_{R5} = k_5^0 (c_{H^+})^n

        .. math::
            \\hat{k}_{TP} = k_{tp}^0 \\hat{c}_h c_c \\exp\\left(\\alpha_{tp}\\frac{F\\hat{\\phi}_c}{RT}\\hat{\\phi}_{fs}\\right)

        .. math::
            \\hat{k}_{O2} = k_{o2}^0 \\exp\\left(\\alpha_{o2}\\frac{2F\\hat{\\phi}_c}{RT} \\left(\\hat{\\phi}_{ext} - \\hat{\\phi}_{o2,eq}\\right) \\right)

        where:
        - :math:`\\hat{\\phi}_{mf}` is the dimensionless potential at metal/film interface
        - :math:`\\hat{\\phi}_{fs}` is the dimensionless potential at film/solution interface
        - :math:`\\alpha_i, \\beta_i` are charge transfer coefficients

        Args:
            t: Time tensor (dimensionless)
            E: Applied potential tensor
            networks: NetworkManager instance
            single: Whether computing for single point or batch

        Returns:
            Tuple of rate constants (k1, k2, k3, k4, k5, ktp, ko2)
        """
        if single:
            batch_size = 1
            x_mf = torch.zeros(1, 1, device=self.device)
        else:
            batch_size = t.shape[0]
            x_mf = torch.zeros(batch_size, 1, device=self.device)

        # Get potentials at interfaces
        inputs_mf = torch.cat([x_mf, t, E], dim=1)
        u_mf = networks['potential'](inputs_mf)  # φ̂_mf

        # Get film thickness
        L_inputs = torch.cat([t, E], dim=1)
        L_pred = networks['film_thickness'](L_inputs)


        x_fs = L_pred

        inputs_fs = torch.cat([x_fs, t, E], dim=1)
        u_fs = networks['potential'](inputs_fs)  # φ̂_fs

        # Compute rate constants using equations above
        F_RT = self.constants.F * self.scales.phic / (self.constants.R * self.constants.T)

        # k₁: Cation vacancy generation at m/f interface
        k1 = self.kinetics.k1_0 * torch.exp(self.kinetics.alpha_cv * 3 * F_RT * u_mf)

        # k₂: Anion vacancy generation at m/f interface
        k2 = self.kinetics.k2_0 * torch.exp(self.kinetics.alpha_av * 2 * F_RT * u_mf)

        # k₃: Cation vacancy consumption at f/s interface
        k3 = self.kinetics.k3_0 * torch.exp(self.kinetics.beta_cv * (3 - self.domain.delta3) * F_RT * u_fs)

        # k₄: Chemical reaction (potential independent)
        k4 = self.kinetics.k4_0

        # k₅: Chemical dissolution
        k5 = self.kinetics.k5_0 * self.materials.c_H

        # k_tp: Hole transfer at f/s interface
        c_h_fs = networks['h'](inputs_fs)
        ktp = self.kinetics.ktp_0 * c_h_fs * self.scales.chc * torch.exp(self.kinetics.alpha_tp * F_RT * u_fs)

        # k_O₂: Oxygen evolution (not used in main equations but included)
        ko2 = self.kinetics.ko2_0 * torch.exp(self.kinetics.a_par * 2 * F_RT * (E - self.kinetics.phi_O2_eq))

        return k1, k2, k3, k4, k5, ktp, ko2

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor, E: torch.Tensor, networks):
        """
        Compute gradients of all network outputs using the gradient computer.

        Args:
            x: Spatial coordinates (requires_grad=True)
            t: Time coordinates (requires_grad=True)
            E: Applied potential
            networks: NetworkManager instance

        Returns:
            GradientResults namedtuple with all gradients
        """
        return self.grad_computer.compute_electrochemistry_gradients(x, t, E, networks)

    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor, E: torch.Tensor, networks):
        """
        Compute PDE residuals for all governing equations.

        **Cation Vacancy Conservation (Dimensionless Nernst-Planck):**

        .. math::
            \\frac{\\partial \\hat{c}_{cv}}{\\partial \\hat{t}} =
            \\frac{D_{cv}\\hat{t}_c}{\\hat{L}_c^2}\\frac{\\partial^2 \\hat{c}_{cv}}{\\partial \\hat{x}^2} +
            \\frac{U_{cv}\\hat{t}_c\\hat{\\phi}_c}{\\hat{L}_c^2}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}}\\frac{\\partial \\hat{c}_{cv}}{\\partial \\hat{x}} +
            \\frac{U_{cv}\\hat{t}_c\\hat{\\phi}_c}{\\hat{L}_c^2}\\hat{c}_{cv}\\frac{\\partial^2 \\hat{\\phi}}{\\partial \\hat{x}^2}

        **Anion Vacancy Conservation (Dimensionless Nernst-Planck):**

        .. math::
            \\frac{\\partial \\hat{c}_{av}}{\\partial \\hat{t}} =
            \\frac{D_{av}\\hat{t}_c}{\\hat{L}_c^2}\\frac{\\partial^2 \\hat{c}_{av}}{\\partial \\hat{x}^2} +
            \\frac{U_{av}\\hat{t}_c\\hat{\\phi}_c}{\\hat{L}_c^2}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}}\\frac{\\partial \\hat{c}_{av}}{\\partial \\hat{x}} +
            \\frac{U_{av}\\hat{t}_c\\hat{\\phi}_c}{\\hat{L}_c^2}\\hat{c}_{av}\\frac{\\partial^2 \\hat{\\phi}}{\\partial \\hat{x}^2}

        **Hole Conservation (Quasi-Steady State):**

        .. math::
            0 = \\frac{D_h\\hat{c}_{h,c}}{\\hat{L}_c^2}\\frac{\\partial^2 \\hat{c}_h}{\\partial \\hat{x}^2} +
            \\frac{FD_h\\hat{\\phi}_c\\hat{c}_{h,c}}{RT\\hat{L}_c^2}\\frac{\\partial \\hat{\\phi}}{\\partial \\hat{x}}\\frac{\\partial \\hat{c}_h}{\\partial \\hat{x}} +
            \\frac{FD_h\\hat{\\phi}_c\\hat{c}_{h,c}}{RT\\hat{L}_c^2}\\hat{c}_h\\frac{\\partial^2 \\hat{\\phi}}{\\partial \\hat{x}^2}

        **Poisson's Equation (Dimensionless):**

        .. math::
            \\frac{\\partial^2 \\hat{\\phi}}{\\partial \\hat{x}^2} =
            -\\frac{F\\hat{L}_c^2\\hat{c}_c}{\\hat{\\phi}_c\\varepsilon_f}\\left(z_{av}\\hat{c}_{av} + z_{cv}\\hat{c}_{cv}\\right)

        Args:
            x: Spatial coordinates (dimensionless)
            t: Time coordinates (dimensionless)
            E: Applied potential
            networks: NetworkManager instance

        Returns:
            Tuple of residuals: (cv_residual, av_residual, h_residual, poisson_residual)
        """
        # Get all gradients using the gradient computer
        grads = self.compute_gradients(x, t, E, networks)

        # Cation vacancy residual - implements equation above
        cv_residual = (grads.c_cv_t -
                       (self.transport.D_cv * self.scales.tc / self.scales.lc ** 2) * grads.c_cv_xx -
                       (self.transport.U_cv * self.scales.tc * self.scales.phic / self.scales.lc ** 2) *
                       grads.phi_x * grads.c_cv_x -
                       (self.transport.U_cv * self.scales.tc * self.scales.phic / self.scales.lc ** 2) *
                       grads.c_cv * grads.phi_xx)

        # Anion vacancy residual - implements equation above
        av_residual = (grads.c_av_t -
                       (self.transport.D_av * self.scales.tc / self.scales.lc ** 2) * grads.c_av_xx -
                       (self.transport.U_av * self.scales.tc * self.scales.phic / self.scales.lc ** 2) *
                       grads.phi_x * grads.c_av_x -
                       (self.transport.U_av * self.scales.tc * self.scales.phic / self.scales.lc ** 2) *
                       grads.c_av * grads.phi_xx)

        # Hole residual (quasi-steady state) - implements equation above
        h_residual = (-(self.transport.D_h * self.scales.chc / self.scales.lc ** 2) * grads.c_h_xx -
                      (self.constants.F * self.transport.D_h * self.scales.phic * self.scales.chc /
                       (self.constants.R * self.constants.T * self.scales.lc ** 2)) *
                      grads.phi_x * grads.c_h_x -
                      (self.constants.F * self.transport.D_h * self.scales.phic * self.scales.chc /
                       (self.constants.R * self.constants.T * self.scales.lc ** 2)) *
                      grads.c_h * grads.phi_xx)

        # Poisson residual - implements equation above
        poisson_residual = (grads.phi_xx +
                            (self.constants.F * self.scales.lc ** 2 * self.scales.cc /
                             (self.scales.phic * self.materials.epsilonf)) *
                            (self.transport.z_av * grads.c_av + self.transport.z_cv * grads.c_cv))

        return cv_residual, av_residual, h_residual, poisson_residual

    def compute_film_growth_rate(self, t: torch.Tensor, E: torch.Tensor, networks):
        """
        Compute film growth rate from electrochemical kinetics.

        **Film Growth Equation:**

        .. math::
            \\frac{dL}{dt} = \\Omega (k_2 - k_5)

        where:
        - :math:`\\Omega` is the molar volume [m³/mol]
        - :math:`k_2` is the anion incorporation rate [mol/(m²·s)]
        - :math:`k_5` is the chemical dissolution rate [mol/(m²·s)]

        **Dimensionless Form:**

        .. math::
            \\frac{d\\hat{L}}{d\\hat{t}} = \\frac{\\hat{t}_c \\Omega}{\\hat{L}_c} (k_2 - k_5)

        Args:
            t: Time tensor (dimensionless)
            E: Applied potential tensor
            networks: NetworkManager instance

        Returns:
            Film growth rate dL/dt (dimensionless)
        """
        # Get rate constants using equations from compute_rate_constants
        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t, E, networks)

        # Film growth rate: dL/dt = Ω(k₂ - k₅)
        dL_dt_dimensionless = (self.scales.tc * self.materials.Omega / self.scales.lc) * (k2 - k5)

        return dL_dt_dimensionless