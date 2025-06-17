import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import torch.onnx
import onnxruntime as ort
from collections import namedtuple

torch.manual_seed(995)#995 is the number stamped onto my necklace

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class FFN(nn.Module):
    """Fully Connected Feed Forward Neural Net"""

    def __init__(self, cfg, input_dim=2, output_dim=1,hidden_layers=5,layer_size=20):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = hidden_layers
        self.layer_size = layer_size

        self.activation = Swish()

        # Input layer
        self.input_layer = nn.Linear(input_dim, self.layer_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.layer_size, self.layer_size)
            for _ in range(self.num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(self.layer_size, output_dim)


    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))

        return self.output_layer(x)
    

class Nexpinnacle():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks eaach having inputs x_hat,t_hat and outputs their name sakes but without dimension
        self.potential_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=1,layer_size=self.cfg.arch.potential.layer_size,hidden_layers=self.cfg.arch.potential.hidden_layers).to(self.device)
        # Need to create a seperate net for the concenration of every species of interest
        self.CV_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=1,layer_size=self.cfg.arch.CV.layer_size,hidden_layers=self.cfg.arch.CV.hidden_layers).to(self.device)  # Cation Vacany
        self.AV_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=1,layer_size=self.cfg.arch.AV.layer_size,hidden_layers=self.cfg.arch.AV.hidden_layers).to(self.device)  # Anion Vacancy
        self.h_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=1,layer_size=self.cfg.arch.h.layer_size,hidden_layers=self.cfg.arch.h.hidden_layers).to(self.device)  # Hole
        self.L_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1,layer_size=self.cfg.arch.L.layer_size,hidden_layers=self.cfg.arch.L.hidden_layers).to(self.device)  # Film Thickness


        # Physics constants
        self.F = cfg.pde.physics.F
        self.R = cfg.pde.physics.R
        self.T = cfg.pde.physics.T
        self.k_B = cfg.pde.physics.k_B
        self.eps0 = cfg.pde.physics.eps0
        self.electron_charge = 1.6e19

        # Diffusion coefficients
        self.D_cv = cfg.pde.physics.D_cv
        self.D_av = cfg.pde.physics.D_av
        self.D_h = cfg.pde.physics.D_h

        # Mobility coefficients
        self.U_cv = cfg.pde.physics.U_cv
        self.U_av = cfg.pde.physics.U_av
        self.U_h = cfg.pde.physics.U_h

        # Species charges
        self.z_cv = cfg.pde.physics.z_cv
        self.z_av = cfg.pde.physics.z_av
        self.z_h = cfg.pde.physics.z_h

        # Permittivities
        self.epsilonf = cfg.pde.physics.epsilonf
        self.eps_film = cfg.pde.physics.eps_film
        self.eps_Ddl = cfg.pde.physics.eps_Ddl
        self.eps_dl = cfg.pde.physics.eps_dl
        self.eps_sol = cfg.pde.physics.eps_sol

        # Semiconductor properties
        self.c_h0 = cfg.pde.physics.c_h0
        self.c_e0 = cfg.pde.physics.c_e0
        self.tau = cfg.pde.physics.tau
        self.Nc = cfg.pde.physics.Nc
        self.Nv = cfg.pde.physics.Nv
        self.mu_e0 = cfg.pde.physics.mu_e0
        self.Ec0 = cfg.pde.physics.Ec0
        self.Ev0 = cfg.pde.physics.Ev0

        # Solution properties
        self.c_H = cfg.pde.physics.c_H
        self.pH = cfg.pde.physics.pH
        self.Omega = cfg.pde.physics.Omega

        # Standard rate constants
        self.k1_0 = cfg.pde.rates.k1_0
        self.k2_0 = cfg.pde.rates.k2_0
        self.k3_0 = cfg.pde.rates.k3_0
        self.k4_0 = cfg.pde.rates.k4_0
        self.k5_0 = cfg.pde.rates.k5_0
        self.ktp_0 = cfg.pde.rates.ktp_0
        self.ko2_0 = cfg.pde.rates.ko2_0

        # Charge transfer coefficients
        self.alpha_cv = cfg.pde.rates.alpha_cv
        self.alpha_av = cfg.pde.rates.alpha_av
        self.beta_cv = cfg.pde.rates.beta_cv
        self.beta_av = cfg.pde.rates.beta_av
        self.alpha_tp = cfg.pde.rates.alpha_tp
        self.a_par = cfg.pde.rates.a_par

        # Derived parameters
        self.a_cv = cfg.pde.rates.a_cv
        self.a_av = cfg.pde.rates.a_av
        self.b_cv = cfg.pde.rates.b_cv

        # Equilibrium potentials
        self.phi_O2_eq = cfg.pde.rates.phi_O2_eq

        # Geometric parameters
        self.d_Ddl = cfg.pde.geometry.d_Ddl
        self.d_dl = cfg.pde.geometry.d_dl
        self.L_cell = cfg.pde.geometry.L_cell

        # Chemistry
        self.delta3 = cfg.pde.chemistry.delta3

        # Domain
        self.time_scale = cfg.domain.time.time_scale
        self.L_initial = cfg.domain.initial.L_initial

        #training
        self.best_loss = float('inf')
        self.training_stage = "physics_first"

        #Charecterestic Scales
        self.lc = cfg.pde.scales.lc
        self.tc = self.lc**2/self.D_cv   
        self.phic = self.R*self.T/self.F
        self.cc = cfg.pde.scales.cc
        self.chc = self.c_h0


        # Optimizer
        params = list(self.potential_net.parameters()) + list(self.CV_net.parameters()) + list(self.AV_net.parameters()) + list(self.h_net.parameters()) + list(self.L_net.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=cfg.optimizer.adam.lr,
            betas=cfg.optimizer.adam.betas,
            eps=cfg.optimizer.adam.eps,
            weight_decay=cfg.optimizer.adam.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.scheduler.RLROP.factor,
            patience=cfg.scheduler.RLROP.patience,
            threshold=cfg.scheduler.RLROP.threshold,
            min_lr=cfg.scheduler.RLROP.min_lr,
        )

    def _grad(self, output, input_var):
        """Take Derrivative of output w.r.t input_var"""
        return torch.autograd.grad(
            output, input_var, 
            grad_outputs=torch.ones_like(output),
            create_graph=True, retain_graph=True
        )[0]

    def compute_gradients(self, x, t, E):
        """Compute the gradients of potential and concentration."""
        x.requires_grad_(True)
        t.requires_grad_(True)

        GradientResults = namedtuple('GradientResults', [
        'phi', 'c_cv', 'c_av', 'c_h',           # Predictions
        'c_cv_t', 'c_av_t', 'c_h_t',            # Time derivatives
        'phi_x', 'c_cv_x', 'c_av_x', 'c_h_x',   # First spatial derivatives
        'phi_xx', 'c_cv_xx', 'c_av_xx', 'c_h_xx' # Second spatial derivatives
        ])

        #Named Tuple cleans up indexing issues

        # Forward pass
        inputs = torch.cat([x, t, E], dim=1)
        phi = self.potential_net(inputs)      # φ̂
        c_cv = self.CV_net(inputs)            # ĉ_cv
        c_av = self.AV_net(inputs)            # ĉ_av
        c_h = self.h_net(inputs)              # ĉ_h
        
        # Time derivatives
        c_cv_t = self._grad(c_cv, t)
        c_av_t = self._grad(c_av, t)
        c_h_t = self._grad(c_h, t)
        
        # First spatial derivatives
        phi_x = self._grad(phi, x)
        c_cv_x = self._grad(c_cv, x)
        c_av_x = self._grad(c_av, x)
        c_h_x = self._grad(c_h, x)
        
        # Second spatial derivatives
        phi_xx = self._grad(phi_x, x)
        c_cv_xx = self._grad(c_cv_x, x)
        c_av_xx = self._grad(c_av_x, x)
        c_h_xx = self._grad(c_h_x, x)
        
        return GradientResults(
            phi, c_cv, c_av, c_h,
            c_cv_t, c_av_t, c_h_t,
            phi_x, c_cv_x, c_av_x, c_h_x,
            phi_xx, c_cv_xx, c_av_xx, c_h_xx
        )

    def compute_rate_constants(self,t,E,single=False):
        """Compute the value of the rate constants for each reaction"""
        if single == True:
            # Predict the potential on the m/f (x=0) boundary
            x_mf = torch.zeros(1, 1, device=self.device)  # Single point
            inputs_mf = torch.cat([x_mf, t, E], dim=1)
            u_mf = self.potential_net(inputs_mf)

            # k1 computation: k1 = k1_0 * exp(alpha_cv * 3F/(RT) * φ_mf)
            k1 = self.k1_0 * torch.exp(self.alpha_cv * 3 * (self.F*self.phic / (self.R * self.T)) * u_mf)

            # k2 computation: k2 = k2_0 * exp(alpha_av * 2F/(RT) * φ_mf)
            k2 = self.k2_0 * torch.exp(self.alpha_av*2* (self.F*self.phic/(self.R*self.T)) * u_mf)

            #Predict L to use in calculation rate constants
            L_inputs = torch.cat([t,E],dim=1)
            L_pred = self.L_net(L_inputs)

            # Predict the potential on the f/s (x=L) boundary
            x_fs = L_pred  # This is already [1, 1]
            inputs_fs = torch.cat([x_fs, t, E], dim=1)
            u_fs = self.potential_net(inputs_fs)

            # k3 computation: k3 = k3_0 * exp(beta_cv * (3-δ)F/(RT) * φ_fs)
            k3 = self.k3_0 * torch.exp(self.beta_cv * (3-self.delta3)* (self.F*self.phic/(self.R*self.T))* u_fs)

            # k4 computation: chemical reaction, potential independent
            k4 = self.k4_0

            # k5 computation: k5 = k5_0 * (c_H+)^n, assuming n=1
            k5 = self.k5_0 * self.c_H

            # Compute the concentration of holes at the f/s interface
            c_h_fs = self.h_net(inputs_fs)

            # ktp computation: ktp = ktp_0 * c_h * exp(alpha_tp * F/(RT) * φ_fs)
            ktp = self.ktp_0 * c_h_fs*self.c_h0 * torch.exp(self.alpha_tp * (self.F*self.phic/(self.R*self.T)) * u_fs)

            # ko2 computation: ko2 = ko2_0 * exp(a_par * 2F/(RT) * (E - φ_O2_eq))
            ko2 = self.ko2_0 * torch.exp(self.a_par * 2 * (self.F*self.phic/(self.R*self.T))* (E - self.phi_O2_eq)) #Do not use this equation, it is certainly incorrect

            return k1, k2, k3, k4, k5, ktp, ko2
        else:
            # Predict the potential on the m/f (x=0) boundary
            x_mf = torch.zeros(t.shape[0], 1, device=self.device)
            inputs_mf = torch.cat([x_mf, t, E], dim=1)
            u_mf = self.potential_net(inputs_mf)

            # k1 computation: k1 = k1_0 * exp(alpha_cv * 3F/(RT) * φ_mf)
            k1 = self.k1_0 * torch.exp(self.alpha_cv * 3 * (self.F*self.phic / (self.R * self.T)) * u_mf)

            # k2 computation: k2 = k2_0 * exp(alpha_av * 2F/(RT) * φ_mf)
            k2 = self.k2_0 * torch.exp(self.alpha_av*2* (self.F*self.phic/(self.R*self.T)) * u_mf)

            #Predict L to use in calculation rate constants
            L_inputs = torch.cat([t,E],dim=1)
            L_pred = self.L_net(L_inputs)

            # Predict the potential on the f/s (x=L) boundary
            x_fs = torch.ones(t.shape[0], 1, device=self.device) * L_pred
            inputs_fs = torch.cat([x_fs, t, E], dim=1)
            u_fs = self.potential_net(inputs_fs)

            # k3 computation: k3 = k3_0 * exp(beta_cv * (3-δ)F/(RT) * φ_fs)
            k3 = self.k3_0 * torch.exp(self.beta_cv * (3-self.delta3)* (self.F*self.phic/(self.R*self.T))* u_fs)

            # k4 computation: chemical reaction, potential independent
            k4 = self.k4_0

            # k5 computation: k5 = k5_0 * (c_H+)^n, assuming n=1
            k5 = self.k5_0 * self.c_H

            # Compute the concentration of holes at the f/s interface
            c_h_fs = self.h_net(inputs_fs)

            # ktp computation: ktp = ktp_0 * c_h * exp(alpha_tp * F/(RT) * φ_fs)
            ktp = self.ktp_0 * c_h_fs*self.c_h0 * torch.exp(self.alpha_tp * (self.F*self.phic/(self.R*self.T)) * u_fs)

            # ko2 computation: ko2 = ko2_0 * exp(a_par * 2F/(RT) * (E_ext - φ_O2_eq))
            ko2 = self.ko2_0 * torch.exp(self.a_par * 2 * (self.F*self.phic/(self.R*self.T))* (E - self.phi_O2_eq)) #Do not use this equation, it is certainly incorrect

            return k1, k2, k3, k4, k5, ktp, ko2
        

    def pde_residuals(self, x, t, E):
        """Compute the residuals due to every PDE"""
        u_pred, cv_pred, av_pred, h_pred, cv_t, av_t,h_t, u_x, cv_x, av_x,h_x, u_xx, cv_xx, av_xx, h_xx = self.compute_gradients(x,t, E)

        # Convection-Diffusion Formulation of Nersnt-Planck
        cd_cv_residual = cv_t + (-(self.D_cv*self.tc/self.lc**2) * cv_xx) + (-(self.U_cv*self.tc*self.phic/self.lc**2) * u_x * cv_x) - ((self.U_cv*self.tc*self.phic/self.lc**2) * cv_pred * u_xx)

        cd_av_residual = av_t + (-(self.D_av*self.tc/self.lc**2) * av_xx) + (-(self.U_av*self.tc*self.phic/self.lc**2) * u_x * av_x) - ((self.U_av*self.tc*self.phic/self.lc**2) * av_pred * u_xx)

        cd_h_residual = (-(self.D_h*self.c_h0/self.lc**2) * h_xx) + (-self.F * self.D_h*self.phic*self.c_h0 * (1 / (self.R * self.T*self.lc**2)) * u_x * h_x) - (self.F * self.D_h * self.phic*self.c_h0 (1 / (self.R * self.T*self.lc**2)) * h_pred * u_xx)  # Using quasai steady state assumption

        # Poisson Residual Calculation

        poisson_residual = u_xx + (self.F*self.lc**2*self.cc*(1/self.phic*self.epsilonf) * (self.z_av * av_pred + self.z_cv * cv_pred))

        return cd_cv_residual, cd_av_residual, cd_h_residual, poisson_residual

    def L_loss(self):
        """Compute the loss of film growth"""
        t = torch.rand(self.cfg.batch_size.L,1,device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min #Pick one random E in the range
        E = single_E.expand(self.cfg.batch_size.L,1) #Broadcast E value to same size as L values
        inputs = torch.cat([t,E],dim=1)
        L_pred = self.L_net(inputs)

        # Get rate constants (using predicted L for f/s boundary)

        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t,E)

        dl_dt_pred = self._grad(L_pred,t)

        dL_dt_physics = (1/self.lc)*self.tc*self.Omega * (k2 - k5)

        return torch.mean((dl_dt_pred-dL_dt_physics)**2)
    

    def initial_condition_loss(self):
        """Compute initial condition losses with individual tracking"""
        t = torch.zeros(self.cfg.batch_size.IC, 1, device=self.device)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.IC,1)

        L_initial_pred = self.get_L_value(t,E)
        x = torch.rand(self.cfg.batch_size.IC, 1, device=self.device) * L_initial_pred 
        t.requires_grad_(True)
        inputs = torch.cat([x, t,E], dim=1)

        L_initial_loss = torch.mean((L_initial_pred-self.L_initial/self.lc)**2)

        # Cation Vacancy Initial Conditions
        cv_initial_pred = self.CV_net(inputs)
        cv_initial_t = self._grad(cv_initial_pred,t)
        cv_initial_loss = torch.mean(cv_initial_pred ** 2) + torch.mean(cv_initial_t ** 2)

        # Anion Vacancy Initial Conditions
        av_initial_pred = self.AV_net(inputs)
        av_initial_t = self._grad(av_initial_pred,t)
        av_initial_loss = torch.mean(av_initial_pred ** 2) + torch.mean(av_initial_t ** 2)

        # Potential Initial Conditions
        u_initial_pred = self.potential_net(inputs)
        u_initial_t = self._grad(u_initial_pred,t)
        poisson_initial_loss = torch.mean((u_initial_pred - ((self.E_ext/self.phic) - 1e7 * (self.lc/self.phic)*x)) ** 2) + torch.mean(u_initial_t ** 2)

        # Hole Initial Conditions
        h_initial_pred = self.h_net(inputs)
        h_initial_t = self._grad(h_initial_pred,t)
        h_initial_loss = torch.mean((h_initial_pred - self.c_h0/self.c_h0) ** 2) + torch.mean(h_initial_t ** 2)


        total_initial_loss = cv_initial_loss + av_initial_loss + poisson_initial_loss + h_initial_loss + L_initial_loss

        return total_initial_loss, cv_initial_loss, av_initial_loss, poisson_initial_loss, h_initial_loss, L_initial_loss


    def boundary_loss(self):
        """Compute boundary losses with individual tracking"""

        t = torch.ones(self.cfg.batch_size.BC,1,device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.BC,1)
        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t,E)
        # m/f interface conditions
        x_mf = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        x_mf.requires_grad_(True)

        #Predict L and compute derrivative for boundary fluxes
        L_input = torch.cat([t,E],dim=1)
        L_pred = self.potential_net(L_input)
        L_pred_t = self._grad(L_pred,t)

        inputs_mf = torch.cat([x_mf, t,E], dim=1)
        # Predicting the potential at m/f interface
        u_pred_mf = self.potential_net(inputs_mf)
        u_pred_mf_x = self._grad(u_pred_mf,x_mf)

        #cv at m/f conditions
        cv_pred_mf = self.CV_net(inputs_mf)
        cv_pred_mf_x = self._grad(cv_pred_mf,x_mf)
        cv_mf_residual = (-self.D_cv*self.cc/self.lc)*cv_pred_mf_x - self.k1_0*torch.exp(self.alpha_cv*self.phic(E/self.phic-u_pred_mf)) - (self.U_cv*self.phic/self.lc*u_pred_mf_x-self.lc/self.tc*L_pred_t)*cv_pred_mf
        cv_mf_loss = torch.mean(cv_mf_residual**2)

        #av at m/f conditions
        av_pred_mf = self.AV_net(inputs_mf)
        av_pred_mf_x = self._grad(av_pred_mf,x_mf)
        av_mf_residual = (-self.D_av*self.cc/self.lc)*av_pred_mf_x - (4/3)*self.k2_0*torch.exp(self.alpha_av*self.phic(E/self.phic-u_pred_mf)) - (self.U_av*self.phic/self.lc*u_pred_mf_x-self.lc/self.tc*L_pred_t)*av_pred_mf
        av_mf_loss = torch.mean(av_mf_residual**2)

        #potential at m/f conditions
        u_mf_residual = (self.eps_film*self.phic/self.lc * u_pred_mf_x) - self.eps_Ddl*self.phic(u_pred_mf-E/self.phic)/self.d_Ddl
        u_mf_loss = torch.mean(u_mf_residual**2)

        # f/s interface conditions
        x_fs = torch.ones(self.cfg.batch_size.BC, 1, device=self.device,requires_grad=True)*L_pred
        inputs_fs = torch.cat([x_fs, t,E], dim=1)

        # Predicting the potential at f/s
        u_pred_fs = self.potential_net(inputs_fs)
        u_pred_fs_x = self._grad(u_pred_fs,x_fs)

        