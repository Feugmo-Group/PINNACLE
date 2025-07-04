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

class Pinnacle():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks eaach having inputs x,t and outputs their name sake
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

        self.initialize_L_net()

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

    def initialize_L_net(self):
        """Initialize L_net output bias to log(L_initial)"""
        target_log_L = torch.log10(torch.tensor(self.L_initial))  # -10
        
        with torch.no_grad():
            # Set output bias to target, keep weights small
            self.L_net.output_layer.bias.fill_(target_log_L)
            self.L_net.output_layer.weight.data *= 0.5  # Make output weights smaller

    def compute_gradients(self, x, t, E):
        """Compute the gradients of potential and concentration."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        # Compute Forward pass
        inputs = torch.cat([x, t, E], dim=1)  # puts all the input values together into one big list
        u_pred = self.potential_net(inputs)  # Acts the neural networks on the inputs to compute u and c
        log_cv_pred = self.CV_net(inputs)
        log_av_pred = self.AV_net(inputs)
        log_h_pred = self.h_net(inputs)

        #Convert log predictions to real cv for pde calculations
        cv_pred = torch.pow(10,log_cv_pred)
        av_pred = torch.pow(10,log_av_pred)
        h_pred = torch.pow(10,log_h_pred)

        # Compute all the time derrivatives

        log_cv_t = torch.autograd.grad(
            log_cv_pred, t, grad_outputs=torch.ones_like(log_cv_pred),
            create_graph=True, retain_graph=True
        )[0]
        cv_t = cv_pred * log_cv_t

        log_av_t = torch.autograd.grad(
            log_av_pred, t, grad_outputs=torch.ones_like(log_av_pred),
            create_graph=True, retain_graph=True
        )[0]
        av_t = av_pred * log_av_t 

        log_h_t = torch.autograd.grad(
            log_h_pred, t, grad_outputs=torch.ones_like(log_h_pred),
            create_graph=True, retain_graph=True
        )[0]
        h_t = log_h_t*h_pred

        # Compute all the derrivatives we need w.r.t x (c derrivatives w.r.t calculated once here and then second calculated when we calculate the flux)
        u_x = torch.autograd.grad(
            u_pred, x, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]

        log_cv_x = torch.autograd.grad(
            log_cv_pred, x, grad_outputs=torch.ones_like(log_cv_pred),
            create_graph=True, retain_graph=True
        )[0]

        cv_x = cv_pred*log_cv_x

        log_av_x = torch.autograd.grad(
            log_av_pred, x, grad_outputs=torch.ones_like(log_av_pred),
            create_graph=True, retain_graph=True
        )[0]

        av_x = av_pred*log_av_x

        log_h_x = torch.autograd.grad(
            log_h_pred, x, grad_outputs=torch.ones_like(log_h_pred),
            create_graph=True, retain_graph=True
        )[0]

        h_x = h_pred*log_h_x

        # Second Derrivatives
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        log_cv_xx = torch.autograd.grad(
            log_cv_x, x, grad_outputs=torch.ones_like(log_cv_x),
            create_graph=True, retain_graph=True
        )[0]

        cv_xx = cv_pred*(log_cv_x**2+log_cv_xx)

        log_av_xx = torch.autograd.grad(
            log_av_x, x, grad_outputs=torch.ones_like(log_av_x),
            create_graph=True, retain_graph=True
        )[0]

        av_xx = av_pred*(log_av_x**2+log_av_xx)

        log_h_xx = torch.autograd.grad(
            log_h_x, x, grad_outputs=torch.ones_like(log_h_x),
            create_graph=True, retain_graph=True
        )[0]

        h_xx = h_pred*(log_h_x**2+log_h_xx)

        return u_pred, cv_pred, av_pred, h_pred, cv_t, av_t, h_t, u_x, cv_x, av_x, h_x, u_xx, cv_xx, av_xx, h_xx
    
    def pde_residuals(self, x, t, E):
        """Compute the residuals due to every PDE"""
        u_pred, cv_pred, av_pred, h_pred, cv_t, av_t,h_t, u_x, cv_x, av_x,h_x, u_xx, cv_xx, av_xx, h_xx = self.compute_gradients(x,t, E)

        # Convection-Diffusion Formulation of Nersnt-Planck
        cd_cv_residual = cv_t + (-self.D_cv * cv_xx) + (-self.U_cv * u_x * cv_x) - (self.U_cv * cv_pred * u_xx)

        cd_av_residual = av_t + (-self.D_av * av_xx) + (-self.U_av * u_x * av_x) - (self.U_av * av_pred * u_xx)

        cd_h_residual = h_t + (-self.D_h * h_xx) + (-self.F * self.D_h * (1 / (self.R * self.T)) * u_x * h_x) - (self.F * self.D_h * (1 / (self.R * self.T)) * h_pred * u_xx)  # Different from ion convection-diffusion, we are ignoring recombination terms as a simpllifying assumtpion

        # Poisson Residual Calculation

        poisson_residual = -self.epsilonf * u_xx - (self.F * (self.z_av * av_pred + self.z_cv * cv_pred))

        return cd_cv_residual, cd_av_residual, cd_h_residual, poisson_residual
    
    def L_loss(self):
        """Enforce dL/dt = Ω(k2 - k5)"""
        if self.training_stage == "physics_first":
            # Stage 1: L is fixed, so dL/dt should be 0
            # Just return a small dummy loss for logging
            return torch.tensor(0.0, device=self.device)
        else:
            t = torch.rand(self.cfg.batch_size.L,1,device=self.device,requires_grad=True) * self.time_scale
            single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min #Pick one random E in the range
            E = single_E.expand(self.cfg.batch_size.L,1) #Broadcast E value to same size as L values
            inputs = torch.cat([t,E],dim=1)
            L_log = self.L_net(inputs)
            
            dL_log_dt = torch.autograd.grad(L_log,t,grad_outputs=torch.ones_like(L_log),create_graph=True,retain_graph=True)[0]

            L_pred = torch.pow(10,L_log)

            dL_dt = L_pred * dL_log_dt 

            # Get rate constants (using predicted L for f/s boundary)

            k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t,E)

            dL_dt_physics = self.Omega * (k2 - k5)
            self._L_diagnostics = {
            'k2_mean': k2.mean().detach().item(),
            'k5_mean': k5, 
            'k2_k5_diff': (k2-k5).mean().detach().item(),
            'omega_k2_k5': (self.Omega*(k2-k5)).mean().detach().item(),
            'dL_dt_pred': dL_dt.mean().detach().item(),
            'dL_dt_physics': dL_dt_physics.mean().detach().item(),
            'L_current': L_pred.mean().detach().item(),
            'log_L_current': L_log.mean().detach().item()
            }

            return torch.mean((dL_dt - dL_dt_physics)**2)
    
    def get_L_value(self,t,E):
        """Get L value - either fixed or from network based on training stage"""

        if self.training_stage == "physics_first":
            # Fix L at initial value
            batch_size = t.shape[0]
            return torch.full((batch_size, 1), self.L_initial, device=self.device)
        else:
        # Use network prediction
            L_inputs = torch.cat([t, E], dim=1)
            L_log = self.L_net(L_inputs)
            return torch.pow(10, L_log)

    def compute_rate_constants(self,t,E,single=False):
        """Compute the value of the rate constants for each reaction"""
        if single == True:
            # Predict the potential on the m/f (x=0) boundary
            x_mf = torch.zeros(1, 1, device=self.device)  # Single point
            inputs_mf = torch.cat([x_mf, t, E], dim=1)
            u_mf = self.potential_net(inputs_mf)

            # k1 computation: k1 = k1_0 * exp(alpha_cv * 3F/(RT) * φ_mf)
            k1 = self.k1_0 * torch.exp(self.alpha_cv * 3 * (self.F / (self.R * self.T)) * u_mf)

            # k2 computation: k2 = k2_0 * exp(alpha_av * 2F/(RT) * φ_mf)
            k2 = self.k2_0 * torch.exp(self.a_cv * u_mf)

            #Predict L to use in calculation rate constants
            L_pred = self.get_L_value(t,E)

            # Predict the potential on the f/s (x=L) boundary
            x_fs = L_pred  # This is already [1, 1]
            inputs_fs = torch.cat([x_fs, t, E], dim=1)
            u_fs = self.potential_net(inputs_fs)

            # k3 computation: k3 = k3_0 * exp(beta_cv * (3-δ)F/(RT) * φ_fs)
            k3 = self.k3_0 * torch.exp(self.b_cv* u_fs)

            # k4 computation: chemical reaction, potential independent
            k4 = self.k4_0

            # k5 computation: k5 = k5_0 * (c_H+)^n, assuming n=1
            k5 = self.k5_0 * self.c_H

            # Compute the concentration of holes at the f/s interface
            c_h_fs = self.h_net(inputs_fs)

            # ktp computation: ktp = ktp_0 * c_h * exp(alpha_tp * F/(RT) * φ_fs)
            ktp = self.ktp_0 * c_h_fs * torch.exp(self.alpha_tp * self.F / (self.R * self.T) * u_fs)

            # ko2 computation: ko2 = ko2_0 * exp(a_par * 2F/(RT) * (E - φ_O2_eq))
            ko2 = self.ko2_0 * torch.exp(self.a_par * 2 * self.F / (self.R * self.T) * (E - self.phi_O2_eq))

            return k1, k2, k3, k4, k5, ktp, ko2
        else:
            # Predict the potential on the m/f (x=0) boundary
            x_mf = torch.zeros(t.shape[0], 1, device=self.device)
            inputs_mf = torch.cat([x_mf, t, E], dim=1)
            u_mf = self.potential_net(inputs_mf)

            # k1 computation: k1 = k1_0 * exp(alpha_cv * 3F/(RT) * φ_mf)
            k1 = self.k1_0 * torch.exp(self.alpha_cv * 3 * self.F / (self.R * self.T) * u_mf)

            # k2 computation: k2 = k2_0 * exp(alpha_av * 2F/(RT) * φ_mf)
            k2 = self.k2_0 * torch.exp(self.a_cv * u_mf)

            #Predict L to use in calculation rate constants
            L_pred = self.get_L_value(t,E)

            # Predict the potential on the f/s (x=L) boundary
            x_fs = torch.ones(t.shape[0], 1, device=self.device) * L_pred
            inputs_fs = torch.cat([x_fs, t, E], dim=1)
            u_fs = self.potential_net(inputs_fs)

            # k3 computation: k3 = k3_0 * exp(beta_cv * (3-δ)F/(RT) * φ_fs)
            k3 = self.k3_0 * torch.exp(self.b_cv* u_fs)

            # k4 computation: chemical reaction, potential independent
            k4 = self.k4_0

            # k5 computation: k5 = k5_0 * (c_H+)^n, assuming n=1
            k5 = self.k5_0 * self.c_H

            # Compute the concentration of holes at the f/s interface
            c_h_fs = self.h_net(inputs_fs)

            # ktp computation: ktp = ktp_0 * c_h * exp(alpha_tp * F/(RT) * φ_fs)
            ktp = self.ktp_0 * c_h_fs * torch.exp(self.alpha_tp * self.F / (self.R * self.T) * u_fs)

            # ko2 computation: ko2 = ko2_0 * exp(a_par * 2F/(RT) * (E_ext - φ_O2_eq))
            ko2 = self.ko2_0 * torch.exp(self.a_par * 2 * self.F / (self.R * self.T) * (E - self.phi_O2_eq))

            return k1, k2, k3, k4, k5, ktp, ko2

    def initial_condition_loss(self):
            """Compute initial condition losses with individual tracking"""
            t_base = torch.zeros(self.cfg.batch_size.IC, 1, device=self.device)
            t_epsilon = torch.rand_like(t_base) * 0.001 * self.time_scale
            t = t_base + t_epsilon
            single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
            E = single_E.expand(self.cfg.batch_size.IC,1)

            L_initial_pred = self.get_L_value(t,E)
            x = torch.rand(self.cfg.batch_size.IC, 1, device=self.device) * L_initial_pred #self.L_initial #Would it be better to change this to L_pred at t=0? vs hard enforcing this?, also we could change the way we are sampling? 
            t.requires_grad_(True)
            inputs = torch.cat([x, t,E], dim=1)

            # Initial Conditions for film thickness
            L_initial_loss = self.cfg.weights.L_initial*torch.mean((L_initial_pred - self.L_initial)**2)

            # Cation Vacancy Initial Conditions
            log_cv_initial_pred = self.CV_net(inputs)
            log_cv_initial_t = torch.autograd.grad(log_cv_initial_pred, t, grad_outputs=torch.ones_like(log_cv_initial_pred), retain_graph=True, create_graph=True)[0]
            cv_initial_pred = torch.pow(10,log_cv_initial_pred)
            threshold = 1e-3
            cv_above_threshold = torch.clamp(cv_initial_pred-threshold,min=0)
            cv_initial_loss = 0.1*torch.mean((cv_above_threshold)** 2) + torch.mean(log_cv_initial_t ** 2) #Equivalent initial conditions in terms of log(c)

            # Anion Vacancy Initial Conditions
            log_av_initial_pred = self.AV_net(inputs)
            log_av_initial_t = torch.autograd.grad(log_av_initial_pred, t, grad_outputs=torch.ones_like(log_av_initial_pred), retain_graph=True, create_graph=True)[0]
            av_initial_pred = torch.pow(10,log_av_initial_pred)
            av_above_threshold = torch.clamp(av_initial_pred-threshold,min=0)
            av_initial_loss = 0.1*torch.mean((av_above_threshold)** 2) + torch.mean(log_av_initial_t ** 2) #Equivalent initial conditions in terms of log(c)

            # Potential Initial Conditions
            u_initial_pred = self.potential_net(inputs)
            u_initial_t = torch.autograd.grad(u_initial_pred, t, grad_outputs=torch.ones_like(u_initial_pred), retain_graph=True, create_graph=True)[0]
            poisson_initial_loss = torch.mean((u_initial_pred - (E-(1e7*x)))**2) + torch.mean(u_initial_t ** 2) #This is very stiff! 
            # Hole Initial Conditions
            log_h_initial_pred = self.h_net(inputs)
            log_h_initial_t = torch.autograd.grad(log_h_initial_pred, t, grad_outputs=torch.ones_like(log_h_initial_pred), retain_graph=True, create_graph=True)[0]
            h_initial_loss = torch.mean((log_h_initial_pred - torch.log10(torch.ones_like(log_h_initial_pred)*self.c_h0)) ** 2) + torch.mean(log_h_initial_t ** 2)

            total_initial_loss = cv_initial_loss + av_initial_loss + poisson_initial_loss + h_initial_loss + L_initial_loss

            return total_initial_loss, cv_initial_loss, av_initial_loss, poisson_initial_loss, h_initial_loss, L_initial_loss
    
    def boundary_loss(self):
        """Compute boundary losses with individual tracking"""

        t = torch.ones(self.cfg.batch_size.BC,1,device=self.device,requires_grad=True)*self.time_scale
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.BC,1)
        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t,E)
        # m/f interface conditions
        x_mf = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        x_mf.requires_grad_(True)

        inputs_mf = torch.cat([x_mf, t,E], dim=1)
        # Predicting the potential at m/f interface
        u_pred_mf = self.potential_net(inputs_mf)
        u_pred_mf_x = torch.autograd.grad(u_pred_mf, x_mf, grad_outputs=torch.ones_like(u_pred_mf), retain_graph=True, create_graph=True)[0]

        # cv at m/f conditions
        log_cv_pred_mf = self.CV_net(inputs_mf)
        log_cv_pred_mf_x = torch.autograd.grad(log_cv_pred_mf, x_mf, grad_outputs=torch.ones_like(log_cv_pred_mf), retain_graph=True, create_graph=True)[0] 
        cv_pred_mf = torch.pow(10,log_cv_pred_mf)
        cv_pred_mf_x = cv_pred_mf*log_cv_pred_mf_x
        #Predict L to use in caclulating BC's
        L_pred = self.get_L_value(t,E)
        
        q1 = self.k1_0* torch.exp(self.alpha_cv*(E-u_pred_mf)) + self.U_cv*u_pred_mf_x  - (self.Omega*(k2-k5)) #analytic enforcing might be stiff
        cv_mf_loss = torch.mean((-self.D_cv*cv_pred_mf_x +q1*cv_pred_mf)**2)

        # av at m/f conditions 
        log_av_pred_mf = self.AV_net(inputs_mf)
        log_av_pred_mf_x = torch.autograd.grad(log_av_pred_mf, x_mf, grad_outputs=torch.ones_like(log_av_pred_mf), retain_graph=True, create_graph=True)[0] 
        av_pred_mf = torch.pow(10,log_av_pred_mf)
        av_pred_mf_x = av_pred_mf*log_av_pred_mf_x
        g2 = (4/3)*self.k2_0*torch.exp(self.alpha_av*(E-u_pred_mf))
        q2 = -1*self.U_av*u_pred_mf_x - (self.Omega*(k2-k5)) #analytic enforcing might be stiff

        av_mf_loss = torch.mean((self.D_av*av_pred_mf_x -g2 +q2*av_pred_mf)**2)

        # potential at m/f conditions
        g3 = self.eps_Ddl* ((u_pred_mf-E)/self.d_Ddl)

        u_mf_loss = torch.mean((-self.epsilonf*u_pred_mf_x -g3)**2)

        # f/s interface conditions
        x_fs = torch.ones(self.cfg.batch_size.BC, 1, device=self.device,requires_grad=True)*L_pred
        inputs_fs = torch.cat([x_fs, t,E], dim=1)
        
        # Predicting the potential at f/s
        u_pred_fs = self.potential_net(inputs_fs)
        u_pred_fs_x = torch.autograd.grad(u_pred_fs, x_fs, grad_outputs=torch.ones_like(u_pred_fs), retain_graph=True, create_graph=True)[0]

        # cv at f/s conditions
        log_cv_pred_fs = self.CV_net(inputs_fs)
        log_cv_pred_fs_x = torch.autograd.grad(log_cv_pred_fs, x_fs, grad_outputs=torch.ones_like(log_cv_pred_fs), retain_graph=True, create_graph=True)[0] 
        cv_pred_fs = torch.pow(10,log_cv_pred_fs)
        cv_pred_fs_x = cv_pred_fs*log_cv_pred_fs_x

        g4 = -1*self.k3_0*torch.exp(self.beta_cv*u_pred_fs)
        q4 = -1*self.U_cv*u_pred_fs_x
        
        cv_fs_loss = torch.mean((-self.D_cv*cv_pred_fs_x -g4 + q4*cv_pred_fs)**2)

        # av at f/s conditions
        log_av_pred_fs = self.AV_net(inputs_fs)
        log_av_pred_fs_x = torch.autograd.grad(log_av_pred_fs, x_fs, grad_outputs=torch.ones_like(log_av_pred_fs), retain_graph=True, create_graph=True)[0] 
        av_pred_fs = torch.pow(10,log_av_pred_fs)
        av_pred_fs_x = av_pred_fs*log_av_pred_fs_x

        q5 = -1*(self.k4_0*torch.exp(self.alpha_av*u_pred_fs) + self.U_av*u_pred_fs_x)
        
        av_fs_loss = torch.mean((-self.D_av*av_pred_fs_x + q5*av_pred_fs)**2)

        # hole at f/s conditions
        log_h_pred_fs = self.h_net(inputs_fs)
        log_h_pred_fs_x = torch.autograd.grad(log_h_pred_fs, x_fs, grad_outputs=torch.ones_like(log_h_pred_fs), retain_graph=True, create_graph=True)[0] 
        h_pred_fs = torch.pow(10,log_h_pred_fs)
        h_pred_fs_x = h_pred_fs*log_h_pred_fs_x

        hole_threshold = 1e-9
        mask = h_pred_fs > hole_threshold
        g6 = torch.zeros_like(h_pred_fs)
        q6 = torch.where(mask, (self.ktp_0 + (self.F*self.D_h)/(self.R*self.T)*u_pred_fs_x), torch.zeros_like(h_pred_fs)) #Might also be very stiff, maybe change? 

        h_fs_loss = torch.mean((-self.D_h*h_pred_fs_x -g6 +q6*h_pred_fs)**2)

        # Potential at f/s loss
        r = -self.d_Ddl*(self.eps_film/self.eps_Ddl)*u_pred_fs_x - self.d_dl*(self.eps_film/self.eps_dl)*u_pred_fs_x
        u_fs_loss = torch.mean((u_pred_fs - r)**2)

        total_BC_loss = cv_mf_loss + av_mf_loss + u_mf_loss + cv_fs_loss + av_fs_loss + u_fs_loss + h_fs_loss

        return total_BC_loss, cv_mf_loss, av_mf_loss, u_mf_loss, cv_fs_loss, av_fs_loss, u_fs_loss, h_fs_loss
    
    def interior_loss(self):
        """Compute PDE residuals on interior points with individual tracking"""
    
        # Sample interior points
        t = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * self.time_scale
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.interior,1)
        L_pred = self.get_L_value(t,E)

        x = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * L_pred 

        # Compute PDE residuals
        cd_cv_residual, cd_av_residual, cd_h_residual, poisson_residual = self.pde_residuals(x, t,E)

        # Calculate individual PDE losses
        cv_pde_loss = torch.mean(cd_cv_residual**2)

        av_pde_loss = torch.mean(cd_av_residual**2)

        h_pde_loss = torch.mean(cd_h_residual**2)

        poisson_pde_loss = torch.mean(poisson_residual**2)

        # Total interior loss
        total_interior_loss = cv_pde_loss + av_pde_loss + h_pde_loss + poisson_pde_loss

        return total_interior_loss, cv_pde_loss, av_pde_loss, h_pde_loss, poisson_pde_loss
    
    def total_loss(self):
        """Compute total weighted loss with detailed breakdown"""
        if self.training_stage == "physics_first":
            # Get detailed losses
            interior_loss, cv_pde_loss, av_pde_loss, h_pde_loss, poisson_pde_loss = self.interior_loss()
            ic_loss, cv_ic_loss, av_ic_loss, poisson_ic_loss, h_ic_loss, L_ic_loss = self.initial_condition_loss()
            bc_loss, cv_mf_loss, av_mf_loss, u_mf_loss, cv_fs_loss, av_fs_loss, u_fs_loss, h_fs_loss = self.boundary_loss()
            L_physics_loss = self.L_loss()
            

            ic_total_no_L = ic_loss - L_ic_loss
            # Apply weights
            total_loss = ((1/self.cfg.batch_size.interior)*interior_loss + 
                        (1/self.cfg.batch_size.BC)*bc_loss + 
                        (1/self.cfg.batch_size.IC)*ic_total_no_L
            )
        
            # Set L losses to zero for logging
            L_physics_loss = torch.tensor(0.0, device=self.device)
            L_ic_loss = torch.tensor(0.0, device=self.device)

        else:
            interior_loss, cv_pde_loss, av_pde_loss, h_pde_loss, poisson_pde_loss = self.interior_loss()
            ic_loss, cv_ic_loss, av_ic_loss, poisson_ic_loss, h_ic_loss, L_ic_loss = self.initial_condition_loss()
            bc_loss, cv_mf_loss, av_mf_loss, u_mf_loss, cv_fs_loss, av_fs_loss, u_fs_loss, h_fs_loss = self.boundary_loss()
            L_physics_loss = self.L_loss()
            

            total_loss = ((1/self.cfg.batch_size.interior)*interior_loss + 
                        (1/self.cfg.batch_size.BC)*bc_loss + 
                        (1/self.cfg.batch_size.IC)*ic_loss+
                        self.cfg.weights.L_physics*(1/self.cfg.batch_size.L)*L_physics_loss
            )


        # Create detailed loss dictionary
        loss_dict = {
            'total': total_loss,
            'interior': (1/self.cfg.batch_size.interior)*interior_loss,
            'boundary': (1/self.cfg.batch_size.BC)*bc_loss,
            'initial': (1/self.cfg.batch_size.IC)*ic_loss,
            'L_physics': self.cfg.weights.L_physics*(1/self.cfg.batch_size.L)*L_physics_loss,
            # PDE losses
            'cv_pde': (1/self.cfg.batch_size.interior)*cv_pde_loss,
            'av_pde': (1/self.cfg.batch_size.interior)*av_pde_loss,
            'h_pde': (1/self.cfg.batch_size.interior)*h_pde_loss,
            'poisson_pde': (1/self.cfg.batch_size.interior)*poisson_pde_loss,
            # Initial condition losses
            'cv_ic': (1/self.cfg.batch_size.IC)*cv_ic_loss,
            'av_ic': (1/self.cfg.batch_size.IC)*av_ic_loss,
            'poisson_ic': (1/self.cfg.batch_size.IC)*poisson_ic_loss,
            'h_ic': (1/self.cfg.batch_size.IC)*h_ic_loss,
            'L_ic': (1/self.cfg.batch_size.IC)*L_ic_loss,
            # Boundary condition losses
            'cv_mf_bc': (1/self.cfg.batch_size.BC)*cv_mf_loss,
            'av_mf_bc': (1/self.cfg.batch_size.BC)*av_mf_loss,
            'u_mf_bc': (1/self.cfg.batch_size.BC)*u_mf_loss,
            'cv_fs_bc': (1/self.cfg.batch_size.BC)*cv_fs_loss,
            'av_fs_bc': (1/self.cfg.batch_size.BC)*av_fs_loss,
            'u_fs_bc': (1/self.cfg.batch_size.BC)*u_fs_loss,
            'h_fs_bc': (1/self.cfg.batch_size.BC)*h_fs_loss
        }
        
        return loss_dict

    def train_step(self):
        """Perform one training step with detailed loss tracking"""
        self.optimizer.zero_grad()
        loss_dict = self.total_loss()
        loss_dict['total'].backward() 

        max_grad_norm = 1.0  # Start with this, adjust if needed
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]['params']], 
            max_norm=max_grad_norm
        )

        self.optimizer.step()
        self.scheduler.step(loss_dict['total'])

        # Convert all losses to float for logging
        loss_dict_float = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
        return loss_dict_float
    
    def load_model(self, path):
        """Load model state from checkpoint"""
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        
        self.potential_net.load_state_dict(checkpoint['potential_net'])
        self.CV_net.load_state_dict(checkpoint['CV_net'])
        self.AV_net.load_state_dict(checkpoint['AV_net'])
        self.h_net.load_state_dict(checkpoint['h_net'])
        self.L_net.load_state_dict(checkpoint['L_net'])
    
    def train(self):
        """Train the model with detailed loss tracking"""
        
        # Initialize loss tracking dictionaries
        loss_history = {
            'total': [], 'interior': [], 'boundary': [], 'initial': [], 'L_physics': [],
            'cv_pde': [], 'av_pde': [], 'h_pde': [], 'poisson_pde': [],
            'cv_ic': [], 'av_ic': [], 'poisson_ic': [], 'h_ic': [], 'L_ic': [],
            'cv_mf_bc': [], 'av_mf_bc': [], 'u_mf_bc': [], 
            'cv_fs_bc': [], 'av_fs_bc': [], 'u_fs_bc': [], 'h_fs_bc': []
        }
        

        # Training loop
        for step in tqdm(range(self.cfg.training.max_steps)):

            if step == self.cfg.training.physics_steps and self.training_stage == "physics_first":
                tqdm.write(f"\n🔄 SWITCHING TO STAGE 2: Training L_net")
                self.training_stage = "full_training"


            loss_dict = self.train_step()
            
            # Store all losses
            for key in loss_history.keys():
                loss_history[key].append(loss_dict[key])
            
            # tqdm.write progress with detailed breakdown
            if step % self.cfg.training.rec_results_freq == 0:
                tqdm.write(f"\n=== Step {step} ===")
                tqdm.write(f"Total Loss: {loss_dict['total']:.6f}")
                tqdm.write(f"Interior: {loss_dict['interior']:.6f} | Boundary: {loss_dict['boundary']:.6f} | "
                    f"Initial: {loss_dict['initial']:.6f} | L_Physics: {loss_dict['L_physics']:.6f}")
                
                tqdm.write("\nPDE Residuals:")
                tqdm.write(f"  CV PDE: {loss_dict['cv_pde']:.6f} | AV PDE: {loss_dict['av_pde']:.6f}")
                tqdm.write(f"  Hole PDE: {loss_dict['h_pde']:.6f} | Poisson PDE: {loss_dict['poisson_pde']:.6f}")
                
                tqdm.write("\nBoundary Conditions:")
                tqdm.write(f"  m/f interface - CV: {loss_dict['cv_mf_bc']:.6f} | AV: {loss_dict['av_mf_bc']:.6f} | U: {loss_dict['u_mf_bc']:.6f}")
                tqdm.write(f"  f/s interface - CV: {loss_dict['cv_fs_bc']:.6f} | AV: {loss_dict['av_fs_bc']:.6f} | U: {loss_dict['u_fs_bc']:.6f} | H: {loss_dict['h_fs_bc']:.6f}")
                
                tqdm.write("\nInitial Conditions:")
                tqdm.write(f"  CV: {loss_dict['cv_ic']:.6f} | AV: {loss_dict['av_ic']:.6f} | H: {loss_dict['h_ic']:.6f}")
                tqdm.write(f"  Poisson: {loss_dict['poisson_ic']:.6f} | L: {loss_dict['L_ic']:.6f}")
                
                if hasattr(self, '_L_diagnostics'):
                    d = self._L_diagnostics
                    tqdm.write(f"\nL Physics Diagnostics:")
                    tqdm.write(f"  k2: {d['k2_mean']:.2e} | k5: {d['k5_mean']:.2e} | (k2-k5): {d['k2_k5_diff']:.2e}")
                    tqdm.write(f"  Ω(k2-k5): {d['omega_k2_k5']:.2e}")
                    tqdm.write(f"  dL/dt predicted: {d['dL_dt_pred']:.2e} | dL/dt physics: {d['dL_dt_physics']:.2e}")
                    tqdm.write(f"  Current L: {d['L_current']:.2e} | log(L): {d['log_L_current']:.2f}")

                current_loss = loss_dict['total']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_checkpoint_path = f"outputs/checkpoints/model_best"
                    self.save_model(self.best_checkpoint_path)
                    
                # Save if specified
                if step % self.cfg.training.save_network_freq == 0 and step > 0:
                    self.save_model(f"outputs/checkpoints_{self.cfg.experiment.name}/model_step_{step}")
                    
                    # Visualize if needed
                    if step % self.cfg.training.rec_inference_freq == 0:
                       self.visualize_predictions(step)
        
        # Final save and tqdm.write
        final_loss = loss_dict
        tqdm.write(f"\n=== Final Results (Step {step}) ===")
        tqdm.write(f"Total Loss: {final_loss['total']:.6f}")
        tqdm.write("PDE Analysis:")
        tqdm.write(f"  Worst PDE: {max([('CV', final_loss['cv_pde']), ('AV', final_loss['av_pde']), ('Hole', final_loss['h_pde']), ('Poisson', final_loss['poisson_pde'])], key=lambda x: x[1])}")
        
        self.save_model(f"outputs/checkpoints_{self.cfg.experiment.name}/model_final")

        if self.best_checkpoint_path:
            tqdm.write(f"🔄 Loading best checkpoint for inference...")
            self.load_model(self.best_checkpoint_path)
            tqdm.write(f"✅ Using best model (loss: {self.best_loss:.6f})")
            
        return loss_history
    
    def plot_potential_profiles(self, step="final"):
        """Plot potential vs position at different times"""
        plots_dir = f"outputs/plots_{self.cfg.experiment.name}"
        
        with torch.no_grad():
            # Fixed potential for comparison
            E_fixed = torch.tensor([[0.8]], device=self.device)
            
            # Three time points: initial, middle, final
            times = [0.0, self.time_scale/2, self.time_scale]
            time_labels = ["t=0 (initial)", f"t={self.time_scale/2:.0f}s (middle)", f"t={self.time_scale:.0f}s (final)"]
            
            plt.figure(figsize=(12, 8))
            
            for i, (t_val, label) in enumerate(zip(times, time_labels)):
                # Get film thickness at this time
                t_tensor = torch.tensor([[t_val]], device=self.device)
                L_current = self.get_L_value(t_tensor, E_fixed)
                L_val = L_current.item()
                
                # Create spatial grid from 0 to L
                n_points = 100
                x_vals = torch.linspace(0, L_val, n_points).to(self.device)
                t_vals = torch.full((n_points, 1), t_val, device=self.device)
                E_vals = torch.full((n_points, 1), 0.8, device=self.device)
                
                # Get potential predictions
                inputs = torch.cat([x_vals.unsqueeze(1), t_vals, E_vals], dim=1)
                u_vals = self.potential_net(inputs)
                reduced_u_vals = (self.electron_charge)*u_vals/(self.k_B*self.T)
                
                # Convert to numpy
                x_np = x_vals.cpu().numpy()
                u_np = u_vals.squeeze().cpu().numpy()
                reduced_u_np = reduced_u_vals.squeeze().cpu().numpy()
                
                # Plot
                plt.subplot(2, 2, i+1)
                plt.plot(x_np * 1e9, u_np, 'b-', linewidth=2,label="Potential")# Convert to nm
                plt.plot(x_np*1e9,reduced_u_np, "r-", linewidth=2,label="Reduced Potential")
                plt.xlabel('Position [nm]')
                plt.ylabel('Potential [V]')
                plt.title(f'{label} (L={L_val*1e9:.1f} nm)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Print interface values
                u_at_0 = u_vals[0].item()
                u_at_L = u_vals[-1].item()
                print(f"{label}: u(0)={u_at_0:.4f}V, u(L)={u_at_L:.4f}V, drop={u_at_0-u_at_L:.4f}V")
            
            # Fourth subplot: Compare all three times
            plt.subplot(2, 2, 4)
            for i, (t_val, label) in enumerate(zip(times, time_labels)):
                t_tensor = torch.tensor([[t_val]], device=self.device)
                L_current = self.get_L_value(t_tensor, E_fixed)
                L_val = L_current.item()
                
                x_vals = torch.linspace(0, L_val, 100).to(self.device)
                t_vals = torch.full((100, 1), t_val, device=self.device)
                E_vals = torch.full((100, 1), 0.8, device=self.device)
                
                inputs = torch.cat([x_vals.unsqueeze(1), t_vals, E_vals], dim=1)
                u_vals = self.potential_net(inputs)
                
                x_np = x_vals.cpu().numpy()
                u_np = u_vals.squeeze().cpu().numpy()
                
                plt.plot(x_np * 1e9, u_np, linewidth=2, label=label)
            
            plt.xlabel('Position [nm]')
            plt.ylabel('Potential [V]')
            plt.title('Potential Evolution Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/potential_profiles_step_{step}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def visualize_predictions(self, step="final"):
        """Visualize network predictions across input ranges - now with E dimension"""
        
        # Create output directory
        plots_dir = f"outputs/plots_{self.cfg.experiment.name}"
        os.makedirs(plots_dir, exist_ok=True)
        self.plot_potential_profiles(step)
        with torch.no_grad():
            # Define input ranges
            n_spatial = 50
            n_temporal = 50
            
            # Fix a representative potential for visualization
            E_fixed = torch.tensor([[0.8]], device=self.device)  # Use middle of range
            
            # Time range (0 to time_scale)
            t_range = torch.linspace(0, self.time_scale, n_temporal).to(self.device)
            
            # Get final film thickness to set spatial range
            t_final = torch.tensor([[float(self.time_scale)]], device=self.device)
            L_final = self.get_L_value(t_final,E_fixed).item()
            x_range = torch.linspace(0, L_final, n_spatial).to(self.device)
            
            tqdm.write(f"Plotting predictions over:")
            tqdm.write(f"  Time range: [0, {self.time_scale:.1f}]")
            tqdm.write(f"  Spatial range: [0, {L_final:.2e}]")
            tqdm.write(f"  Fixed potential: {E_fixed.item():.1f}V")
            
            # Create 2D grid for contour plots
            T_mesh, X_mesh = torch.meshgrid(t_range, x_range, indexing='ij')
            E_mesh = torch.full_like(T_mesh, E_fixed.item())  # Fixed E for all points
            
            # Stack inputs for 3D networks
            inputs_3d = torch.stack([
                X_mesh.flatten(), 
                T_mesh.flatten(), 
                E_mesh.flatten()
            ], dim=1)
            
            # Get 3D network predictions
            u_2d = self.potential_net(inputs_3d).reshape(n_temporal, n_spatial)
            cv_2d = self.CV_net(inputs_3d).reshape(n_temporal, n_spatial)
            av_2d = self.AV_net(inputs_3d).reshape(n_temporal, n_spatial)
            h_2d = self.h_net(inputs_3d).reshape(n_temporal, n_spatial)
            
            # Film thickness evolution (depends on t and E)
            t_1d = t_range.unsqueeze(1)
            E_1d = torch.full_like(t_1d, E_fixed.item())
            L_inputs_1d = torch.cat([t_1d, E_1d], dim=1)
            L_1d = self.L_net(L_inputs_1d).squeeze()
            
            # Convert to numpy
            t_np = t_range.cpu().numpy()
            x_np = x_range.cpu().numpy()
            T_np, X_np = np.meshgrid(t_np, x_np, indexing='ij')
            
            u_np = u_2d.cpu().numpy()
            cv_np = cv_2d.cpu().numpy()
            av_np = av_2d.cpu().numpy()
            h_np = h_2d.cpu().numpy()
            L_np = L_1d.cpu().numpy()
            
            # Create plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Potential field
            im1 = axes[0,0].contourf(X_np, T_np, u_np, levels=20, cmap='RdBu_r')
            axes[0,0].set_xlabel('Position')
            axes[0,0].set_ylabel('Time')
            axes[0,0].set_title(f'Potential φ(x,t) at E={E_fixed.item():.1f}V')
            plt.colorbar(im1, ax=axes[0,0])
            
            # 2. Cation vacancies
            im2 = axes[0,1].contourf(X_np, T_np, cv_np, levels=20, cmap='Reds')
            axes[0,1].set_xlabel('Position')
            axes[0,1].set_ylabel('Time')
            axes[0,1].set_title(f'Log Cation Vacancies at E={E_fixed.item():.1f}V')
            plt.colorbar(im2, ax=axes[0,1])
            
            # 3. Anion vacancies
            im3 = axes[0,2].contourf(X_np, T_np, av_np, levels=20, cmap='Blues')
            axes[0,2].set_xlabel('Position')
            axes[0,2].set_ylabel('Time')
            axes[0,2].set_title(f'Log Anion Vacancies at E={E_fixed.item():.1f}V')
            plt.colorbar(im3, ax=axes[0,2])
            
            # 4. Holes
            im4 = axes[1,0].contourf(X_np, T_np, h_np, levels=20, cmap='Purples')
            axes[1,0].set_xlabel('Position')
            axes[1,0].set_ylabel('Time')
            axes[1,0].set_title(f'Log Holes at E={E_fixed.item():.1f}V')
            plt.colorbar(im4, ax=axes[1,0])
            
            # 5. Film thickness
            axes[1,1].plot(t_np, L_np, 'k-', linewidth=3)
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('Film Thickness')
            axes[1,1].set_title(f'Film Thickness L(t) at E={E_fixed.item():.1f}V')
            axes[1,1].grid(True, alpha=0.3)
            
            # 6. Potential dependence at fixed time/position
            E_sweep = torch.linspace(0.5, 1.5, 50).to(self.device)
            t_mid = torch.full((50, 1), self.time_scale/2, device=self.device)
            x_mid = torch.full((50, 1), L_final/2, device=self.device)
            
            E_sweep_inputs = torch.cat([x_mid, t_mid, E_sweep.unsqueeze(1)], dim=1)
            u_vs_E = self.potential_net(E_sweep_inputs).cpu().numpy()
            
            axes[1,2].plot(E_sweep.cpu().numpy(), u_vs_E, 'r-', linewidth=2)
            axes[1,2].set_xlabel('Applied Potential E [V]')
            axes[1,2].set_ylabel('Film Potential [V]')
            axes[1,2].set_title(f'Potential vs E (x={L_final/2:.1e}, t={self.time_scale/2:.1f})')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Network Predictions Overview - Step {step}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{plots_dir}/predictions_overview_step_{step}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print statistics
            tqdm.write(f"\nPrediction Statistics (Step {step}) at E={E_fixed.item():.1f}V:")
            tqdm.write("-" * 50)
            tqdm.write(f"Potential:        {u_np.min():.2e} to {u_np.max():.2e} (mean: {u_np.mean():.2e})")
            tqdm.write(f"Cation Vacancies: {cv_np.min():.2e} to {cv_np.max():.2e} (mean: {cv_np.mean():.2e})")
            tqdm.write(f"Anion Vacancies:  {av_np.min():.2e} to {av_np.max():.2e} (mean: {av_np.mean():.2e})")
            tqdm.write(f"Holes:            {h_np.min():.2e} to {h_np.max():.2e} (mean: {h_np.mean():.2e})")
            tqdm.write(f"Film Thickness:   {L_np.min():.2e} to {L_np.max():.2e}")

    def save_model(self, name):
        """Save model state."""
        torch.save({
            'potential_net': self.potential_net.state_dict(),
            'CV_net': self.CV_net.state_dict(),  
            'AV_net': self.AV_net.state_dict(),  
            'h_net': self.h_net.state_dict(),
            'L_net': self.L_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f"{name}.pt")


    def export_for_netron(self):
        """Export combined model for Netron visualization - updated for 3D inputs"""
        
        save_path = f"../outputs/pinnacle_architecture.onnx"
        os.makedirs("../outputs", exist_ok=True)
        
        # Create a simple combined model
        class CombinedPINNACLE(torch.nn.Module):
            def __init__(self, potential_net, CV_net, AV_net, h_net, L_net):
                super().__init__()
                # Name them clearly for Netron
                self.potential_network = potential_net
                self.cation_vacancy_network = CV_net
                self.anion_vacancy_network = AV_net
                self.hole_network = h_net
                self.film_thickness_network = L_net
                
            def forward(self, x, t, E):
                # Combine x,t,E for spatial-temporal-potential networks
                xte_input = torch.cat([x, t, E], dim=1)
                
                # Combine t,E for film thickness network
                te_input = torch.cat([t, E], dim=1)
                
                # Get all predictions
                potential = self.potential_network(xte_input)
                cv_conc = self.cation_vacancy_network(xte_input)
                av_conc = self.anion_vacancy_network(xte_input)
                h_conc = self.hole_network(xte_input)
                thickness = self.film_thickness_network(te_input)
                
                return potential, cv_conc, av_conc, h_conc, thickness
        
        # Move to CPU and eval mode
        self.potential_net.to('cpu').eval()
        self.CV_net.to('cpu').eval()
        self.AV_net.to('cpu').eval()
        self.h_net.to('cpu').eval()
        self.L_net.to('cpu').eval()
        
        # Create combined model
        combined = CombinedPINNACLE(self.potential_net, self.CV_net, self.AV_net, self.h_net, self.L_net)
        
        # Export to ONNX
        dummy_x = torch.randn(1, 1)  # spatial position
        dummy_t = torch.randn(1, 1)  # time
        dummy_E = torch.randn(1, 1)  # applied potential
        
        torch.onnx.export(
            combined,
            (dummy_x, dummy_t, dummy_E),
            save_path,
            input_names=['x_position', 't_time', 'E_applied'],
            output_names=['potential', 'cation_vacancy_conc', 'anion_vacancy_conc', 'hole_conc', 'film_thickness']
        )
        
        tqdm.write(f"✅ Architecture exported for 3D inputs (x, t, E)!")

    def generate_polarization_curve(self, n_points=50):
        """Generate polarization curve at specified time"""
        
        t_eval = self.time_scale  # Use final time by default
        
        tqdm.write(f"Generating polarization curve at t={t_eval}")
        E_range = (self.cfg.pde.physics.E_min,self.cfg.pde.physics.E_max)
        with torch.no_grad():
            # Create potential sweep
            E_values = torch.linspace(E_range[0], E_range[1], n_points).to(self.device)
            currents = []
            
            for E_val in E_values:
                # Calculate current at f/s interface
                t_tensor = torch.tensor([[t_eval]], device=self.device)
                E_tensor = torch.tensor([[E_val.item()]], device=self.device)
                
                # Get film thickness at this time and potential
                L_val = self.get_L_value(t_tensor,E_tensor)
                
                # Evaluate at f/s interface (x = L)
                x_fs = L_val
                x_mf = torch.zeros_like(L_val)
                inputs_fs = torch.cat([x_fs, t_tensor, E_tensor], dim=1)
                inputs_mf = torch.cat([x_mf,t_tensor,E_tensor],dim=1)

                # Get concentrations and rate constants
                k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t_tensor, E_tensor,single=True)
                h_fs = torch.pow(10,self.h_net(inputs_fs))  # Convert from log
                av_fs = torch.pow(10,self.AV_net(inputs_fs))
                u_fs = self.potential_net(inputs_fs)
                cv_mf = torch.pow(10,self.CV_net(inputs_mf))
                # Calculate current contributions (mol/(m²·s) -> current density)
                # Convert to A/m² using Faraday constant
                current_k1 = (8.0/3.0) * k1 * self.F * cv_mf
                current_k2 = (8.0/3.0) * self.F *k2
                current_k3 = (1/3) * self.F * k3  # 3 electrons per k3 reaction  
                current_ktp = (-1.0) * self.F * ktp * h_fs  # 1 electron per hole Might need a different handling
                #current_ko2 = 2 * self.F * ko2  # 2 electrons per O2 reaction
                # Total current (can be positive or negative)
                print(f"i_1:{current_k1.item():.6f}")
                print(f"i_2:{current_k2.item():.6f}")
                print(f"i_3:{current_k3.item():.6f}")
                print(f"i_tp:{current_ktp.item():.6f}")

                total_current = current_k1 + current_k2+ current_k3 + current_ktp
                currents.append(total_current.item())
            
            # Convert to numpy for plotting
            E_np = E_values.cpu().numpy()
            I_np = np.array(currents)
            
            # Create polarization curve plot
            plt.figure(figsize=(10, 6))
            plt.plot(E_np, I_np, 'b-', linewidth=2, label='Total Current')
            plt.xlabel('Applied Potential [V]')
            plt.ylabel('Current Density [A/m²]')
            plt.title(f'Polarization Curve at t={t_eval}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plots_dir = f"outputs/plots_{self.cfg.experiment.name}"
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(f"{plots_dir}/polarization_curve_t_{t_eval}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            tqdm.write(f"Current range: {I_np.min():.2e} to {I_np.max():.2e} A/m²")
            
            return E_np, I_np
        

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    tqdm.write(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = Pinnacle(cfg)
    
    # Train with detailed loss tracking
    os.makedirs(f"outputs/checkpoints_{cfg.experiment.name}",exist_ok=True)
    loss_history = model.train()

    # Export to ONNX after training
    tqdm.write("\n" + "="*50)
    tqdm.write("Exporting trained model to ONNX...")
    model.export_for_netron()
    tqdm.write("="*50)

    model.potential_net.to(model.device)
    model.CV_net.to(model.device) 
    model.AV_net.to(model.device)
    model.h_net.to(model.device)
    model.L_net.to(model.device)
    
    if hasattr(model, 'best_checkpoint_path') and model.best_checkpoint_path:
        tqdm.write(f"🔄 Reloading best checkpoint for inference...")
        model.load_model(model.best_checkpoint_path)
        tqdm.write(f"✅ Using best model for polarization curve (loss: {model.best_loss:.6f})")
    
    # Create comprehensive loss plots
    model.visualize_predictions("best")
    plot_detailed_losses(loss_history,cfg.experiment.name)
    E_values, current_values = model.generate_polarization_curve()

def plot_detailed_losses(loss_history,experiment_name):
    """Create comprehensive plots of all loss components"""
    
    # Create output directory
    os.makedirs(f"outputs/plots_{experiment_name}", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Total and main components
    axes[0,0].semilogy(loss_history['total'], label='Total Loss', linewidth=2)
    axes[0,0].semilogy(loss_history['interior'], label='Interior (PDE)', alpha=0.8)
    axes[0,0].semilogy(loss_history['boundary'], label='Boundary', alpha=0.8)
    axes[0,0].semilogy(loss_history['initial'], label='Initial', alpha=0.8)
    axes[0,0].semilogy(loss_history['L_physics'], label='Film Thickness', alpha=0.8)
    axes[0,0].set_title('Main Loss Components')
    axes[0,0].set_xlabel('Training Step')
    axes[0,0].set_ylabel('Loss (log scale)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # PDE residuals breakdown
    axes[0,1].semilogy(loss_history['cv_pde'], label='CV PDE', alpha=0.8)
    axes[0,1].semilogy(loss_history['av_pde'], label='AV PDE', alpha=0.8)
    axes[0,1].semilogy(loss_history['h_pde'], label='Hole PDE', alpha=0.8)
    axes[0,1].semilogy(loss_history['poisson_pde'], label='Poisson PDE', alpha=0.8)
    axes[0,1].set_title('Individual PDE Residuals')
    axes[0,1].set_xlabel('Training Step')
    axes[0,1].set_ylabel('Loss (log scale)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Boundary conditions breakdown
    axes[1,0].semilogy(loss_history['cv_mf_bc'], label='CV (m/f)', alpha=0.8)
    axes[1,0].semilogy(loss_history['av_mf_bc'], label='AV (m/f)', alpha=0.8)
    axes[1,0].semilogy(loss_history['u_mf_bc'], label='Potential (m/f)', alpha=0.8)
    axes[1,0].semilogy(loss_history['cv_fs_bc'], label='CV (f/s)', alpha=0.8)
    axes[1,0].semilogy(loss_history['av_fs_bc'], label='AV (f/s)', alpha=0.8)
    axes[1,0].semilogy(loss_history['u_fs_bc'], label='Potential (f/s)', alpha=0.8)
    axes[1,0].semilogy(loss_history['h_fs_bc'], label='Hole (f/s)', alpha=0.8)
    axes[1,0].set_title('Boundary Condition Losses')
    axes[1,0].set_xlabel('Training Step')
    axes[1,0].set_ylabel('Loss (log scale)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Initial conditions breakdown
    axes[1,1].semilogy(loss_history['cv_ic'], label='CV IC', alpha=0.8)
    axes[1,1].semilogy(loss_history['av_ic'], label='AV IC', alpha=0.8)
    axes[1,1].semilogy(loss_history['h_ic'], label='Hole IC', alpha=0.8)
    axes[1,1].semilogy(loss_history['poisson_ic'], label='Poisson IC', alpha=0.8)
    axes[1,1].semilogy(loss_history['L_ic'], label='Film Thickness IC', alpha=0.8)
    axes[1,1].set_title('Initial Condition Losses')
    axes[1,1].set_xlabel('Training Step')
    axes[1,1].set_ylabel('Loss (log scale)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots_{experiment_name}/detailed_training_losses.png", dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    main()

