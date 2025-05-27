import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import torch.onnx
import onnx
import onnxruntime as ort

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class FFN(nn.Module):
    """Fully Connected Feed Forward Neural Net"""

    def __init__(self, cfg, input_dim=2, output_dim=1):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = cfg.nr_layers
        self.layer_size = cfg.layer_size

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
        x_input = x
        x = self.activation(self.input_layer(x))

        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))

        return self.output_layer(x)


class Pinnacle():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks eaach having inputs x,t and outputs their name sake
        self.potential_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)
        # Need to create a seperate net for the concenration of every species of interest
        self.CV_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)  # Cation Vacany
        self.AV_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)  # Anion Vacancy
        self.h_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)  # Hole
        self.L_net = FFN(cfg.arch.fully_connected, input_dim=1, output_dim=1).to(self.device)  # Film Thickness

        # Physics constants
        self.F = cfg.pde.physics.F
        self.R = cfg.pde.physics.R
        self.T = cfg.pde.physics.T
        self.k_B = cfg.pde.physics.k_B
        self.eps0 = cfg.pde.physics.eps0
        self.E_ext = cfg.pde.physics.E_ext

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

        # Optimizer
        params = list(self.potential_net.parameters()) + list(self.CV_net.parameters()) + list(self.AV_net.parameters()) + list(self.h_net.parameters()) + list(self.L_net.parameters())
        self.optimizer = optim.Adam(
            params,
            lr=cfg.optimizer.adam.lr,
            betas=cfg.optimizer.adam.betas,
            eps=cfg.optimizer.adam.eps,
            weight_decay=cfg.optimizer.adam.weight_decay
        )

        # Scheduler/ Same as in PhysicsNEMO for now
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.scheduler.tf_exponential_lr.decay_steps,
            gamma=cfg.scheduler.tf_exponential_lr.decay_rate
        )

        # Loss weights
        #self.poisson_weight = cfg.pde.weights.poisson_weight
        #self.nernst_weight = cfg.pde.weights.nernst_weight
        #self.bc_weight = cfg.pde.weights.bc_weight

    def compute_gradients(self, x, t):
        """Compute the gradients of potential and concentration."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        # Compute Forward pass
        inputs = torch.cat([x, t], dim=1)  # puts all the input values together into one big list
        u_pred = self.potential_net(inputs)  # Acts the neural networks on the inputs to compute u and c
        cv_pred = self.CV_net(inputs)
        av_pred = self.AV_net(inputs)
        h_pred = self.h_net(inputs)

        # Compute all the time derrivatives
        cv_t = torch.autograd.grad(
            cv_pred, t, grad_outputs=torch.ones_like(cv_pred),
            create_graph=True, retain_graph=True
        )[0]

        av_t = torch.autograd.grad(
            av_pred, t, grad_outputs=torch.ones_like(av_pred),
            create_graph=True, retain_graph=True
        )[0]
        h_t = torch.autograd.grad(
            h_pred, t, grad_outputs=torch.ones_like(h_pred),
            create_graph=True, retain_graph=True
        )[0]

        # Compute all the derrivatives we need w.r.t x (c derrivatives w.r.t calculated once here and then second calculated when we calculate the flux)
        u_x = torch.autograd.grad(
            u_pred, x, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]

        cv_x = torch.autograd.grad(
            cv_pred, x, grad_outputs=torch.ones_like(cv_pred),
            create_graph=True, retain_graph=True
        )[0]

        av_x = torch.autograd.grad(
            av_pred, x, grad_outputs=torch.ones_like(av_pred),
            create_graph=True, retain_graph=True
        )[0]
        h_x = torch.autograd.grad(
            h_pred, x, grad_outputs=torch.ones_like(h_pred),
            create_graph=True, retain_graph=True
        )[0]

        # Second Derrivatives
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        cv_xx = torch.autograd.grad(
            cv_x, x, grad_outputs=torch.ones_like(cv_x),
            create_graph=True, retain_graph=True
        )[0]
        av_xx = torch.autograd.grad(
            av_x, x, grad_outputs=torch.ones_like(av_x),
            create_graph=True, retain_graph=True
        )[0]
        h_xx = torch.autograd.grad(
            h_x, x, grad_outputs=torch.ones_like(h_x),
            create_graph=True, retain_graph=True
        )[0]

        return u_pred, cv_pred, av_pred, h_pred, cv_t, av_t, h_t, u_x, cv_x, av_x, h_x, u_xx, cv_xx, av_xx, h_xx

    # enforce alll the pde losses
    def pde_residuals(self, x, t):
        """Compute the residuals due to every PDE"""
        u_pred, cv_pred, av_pred, h_pred, cv_t, av_t,h_t, u_x, cv_x, av_x,h_x, u_xx, cv_xx, av_xx, h_xx = self.compute_gradients(x, t)

        # Convection-Diffusion Formulation of Nersnt-Planck
        cd_cv_residual = cv_t + (-self.D_cv * cv_xx) + (-self.U_cv * u_x * cv_x) - (self.U_cv * cv_pred * u_xx)

        cd_av_residual = av_t + (-self.D_av * av_xx) + (-self.U_av * u_x * av_x) - (self.U_av * av_pred * u_xx)

        cd_h_residual = h_t + (-self.D_h * h_xx) + (-self.F * self.D_h * (1 / self.R * self.T) * u_x * h_x) - (self.F * self.D_h * (1 / self.R * self.T) * h_pred * u_xx)  # Different from ion convection-diffusion, we are ignoring recombination terms as a simpllifying assumtpion

        # Poisson Residual Calculation

        poisson_residual = -self.epsilonf * u_xx - (self.F * (self.z_av * av_pred + self.z_cv * cv_pred))

        return cd_cv_residual, cd_av_residual, cd_h_residual, poisson_residual

    def L_loss(self):
        """Enforce dL/dt = Ω(k2 - k5)"""
        t = torch.rand(self.cfg.batch_size.film_thickness,1,device=self.device,requires_grad=True) * self.time_scale

        L_pred = self.L_net(t)

        dL_dt = torch.autograd.grad(L_pred,t,grad_outputs=torch.ones_like(L_pred),create_graph=True,retain_graph=True)[0]

        # Get rate constants (using predicted L for f/s boundary)
        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants()

        dL_dt_physics = self.Omega * (k2 - k5)

        return torch.mean( (dL_dt - dL_dt_physics)**2 )


    def get_E_ext(self, t):
        """Dummy function for external potential - will implement sweeping later"""

        return self.E_ext  # For now just return constant value

    def compute_rate_constants(self):
        
        #Initialize prediction time range
        t = torch.rand(self.cfg.batch_size.rate, 1, device=self.device) * self.time_scale
        
        # Predict the potential on the m/f (x=0) boundary
        x_mf = torch.zeros(self.cfg.batch_size.rate, 1, device=self.device)
        inputs_mf = torch.cat([x_mf, t], dim=1)
        u_mf = self.potential_net(inputs_mf)

        # k1 computation: k1 = k1_0 * exp(alpha_cv * 3F/(RT) * φ_mf)
        k1 = self.k1_0 * torch.exp(self.alpha_cv * 3 * self.F / (self.R * self.T) * u_mf)

        # k2 computation: k2 = k2_0 * exp(alpha_av * 2F/(RT) * φ_mf)
        k2 = self.k2_0 * torch.exp(self.alpha_av * 2 * self.F / (self.R * self.T) * u_mf)

        #Predict L to use in calculation rate constants
        L_pred = self.L_net(t)

        # Predict the potential on the f/s (x=L) boundary
        x_fs = torch.ones(self.cfg.batch_size.rate, 1, device=self.device) * L_pred
        inputs_fs = torch.cat([x_fs, t], dim=1)
        u_fs = self.potential_net(inputs_fs)

        # k3 computation: k3 = k3_0 * exp(beta_cv * (3-δ)F/(RT) * φ_fs)
        k3 = self.k3_0 * torch.exp(self.beta_cv * (3 - self.delta3) * self.F / (self.R * self.T) * u_fs)

        # k4 computation: chemical reaction, potential independent
        k4 = self.k4_0

        # k5 computation: k5 = k5_0 * (c_H+)^n, assuming n=1
        k5 = self.k5_0 * self.c_H

        # Compute the concentration of holes at the f/s interface
        c_h_fs = self.h_net(inputs_fs)

        # ktp computation: ktp = ktp_0 * c_h * exp(alpha_tp * F/(RT) * φ_fs)
        ktp = self.ktp_0 * c_h_fs * torch.exp(self.alpha_tp * self.F / (self.R * self.T) * u_fs)

        # ko2 computation: ko2 = ko2_0 * exp(a_par * 2F/(RT) * (E_ext - φ_O2_eq))
        ko2 = self.ko2_0 * np.exp(self.a_par * 2 * self.F / (self.R * self.T) * (self.E_ext - self.phi_O2_eq))

        return k1, k2, k3, k4, k5, ktp, ko2

    def initial_condition_loss(self):
        """Compute initial condition losses with individual tracking"""
        
        t = torch.zeros(self.cfg.batch_size.IC, 1, device=self.device)
        x = torch.rand(self.cfg.batch_size.IC, 1, device=self.device) * self.L_initial
        t.requires_grad_(True)
        inputs = torch.cat([x, t], dim=1)

        # Initial Conditions for film thickness
        L_initial_pred = self.L_net(t)
        L_initial_loss = torch.mean((L_initial_pred - self.L_initial)**2)

        # Cation Vacancy Initial Conditions
        cv_initial_pred = self.CV_net(inputs)
        cv_initial_t = torch.autograd.grad(cv_initial_pred, t, grad_outputs=torch.ones_like(cv_initial_pred), retain_graph=True, create_graph=True)[0]
        cv_initial_loss = torch.mean(cv_initial_pred ** 2) + torch.mean(cv_initial_t ** 2)

        # Anion Vacancy Initial Conditions
        av_initial_pred = self.AV_net(inputs)
        av_initial_t = torch.autograd.grad(av_initial_pred, t, grad_outputs=torch.ones_like(av_initial_pred), retain_graph=True, create_graph=True)[0]
        av_initial_loss = torch.mean(av_initial_pred ** 2) + torch.mean(av_initial_t ** 2)

        # Potential Initial Conditions
        u_initial_pred = self.potential_net(inputs)
        u_initial_t = torch.autograd.grad(u_initial_pred, t, grad_outputs=torch.ones_like(u_initial_pred), retain_graph=True, create_graph=True)[0]
        poisson_initial_loss = torch.mean((u_initial_pred - (self.E_ext - 1e7 * x)) ** 2) + torch.mean(u_initial_t ** 2)

        # Hole Initial Conditions
        h_initial_pred = self.h_net(inputs)
        h_initial_t = torch.autograd.grad(h_initial_pred, t, grad_outputs=torch.ones_like(h_initial_pred), retain_graph=True, create_graph=True)[0]
        h_initial_loss = torch.mean((h_initial_pred - self.c_h0) ** 2) + torch.mean(h_initial_t ** 2)

        total_initial_loss = cv_initial_loss + av_initial_loss + poisson_initial_loss + h_initial_loss + L_initial_loss

        return total_initial_loss, cv_initial_loss, av_initial_loss, poisson_initial_loss, h_initial_loss, L_initial_loss

    def boundary_loss(self):
        """Compute boundary losses with individual tracking"""
        
        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants()

        # Initialize the time inputs
        t = torch.rand(self.cfg.batch_size.BC, 1, device=self.device) * self.time_scale
        
        # m/f interface conditions
        x_mf = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        x_mf.requires_grad_(True)
        inputs_mf = torch.cat([x_mf, t], dim=1)

        # Predicting the potential at m/f interface
        u_pred_mf = self.potential_net(inputs_mf)
        u_pred_mf_x = torch.autograd.grad(u_pred_mf, x_mf, grad_outputs=torch.ones_like(u_pred_mf), retain_graph=True, create_graph=True)[0]

        # cv at m/f conditions
        cv_pred_mf = self.CV_net(inputs_mf)
        cv_pred_mf_x = torch.autograd.grad(cv_pred_mf, x_mf, grad_outputs=torch.ones_like(cv_pred_mf), retain_graph=True, create_graph=True)[0] 

        q1 = self.k1_0* torch.exp(self.alpha_cv*(self.E_ext-u_pred_mf)) + self.U_cv*u_pred_mf_x  - (self.Omega*(k2-k5))
        cv_mf_loss = torch.mean((-self.D_cv*cv_pred_mf_x +q1*cv_pred_mf)**2)

        # av at m/f conditions 
        av_pred_mf = self.AV_net(inputs_mf)
        av_pred_mf_x = torch.autograd.grad(av_pred_mf, x_mf, grad_outputs=torch.ones_like(av_pred_mf), retain_graph=True, create_graph=True)[0] 

        g2 = (4/3)*self.k2_0*torch.exp(self.alpha_av*(self.E_ext-u_pred_mf))
        q2 = -1*self.U_av*u_pred_mf_x - (self.Omega*(k2-k5))

        av_mf_loss = torch.mean((self.D_av*av_pred_mf_x -g2 +q2*av_pred_mf)**2)

        # potential at m/f conditions
        g3 = self.eps_Ddl* ((u_pred_mf-self.E_ext)/self.d_Ddl)
        u_mf_loss = torch.mean((-self.epsilonf*u_pred_mf_x -g3)**2)

        # Predict L to compute the location of the f/s interface
        L_pred = self.L_net(t)

        # f/s interface conditions
        x_fs = L_pred.detach().clone().requires_grad_(True)
        inputs_fs = torch.cat([x_fs, t], dim=1)
        
        # Predicting the potential at f/s
        u_pred_fs = self.potential_net(inputs_fs)
        u_pred_fs_x = torch.autograd.grad(u_pred_fs, x_fs, grad_outputs=torch.ones_like(u_pred_fs), retain_graph=True, create_graph=True)[0]

        # cv at f/s conditions
        cv_pred_fs = self.CV_net(inputs_fs)
        cv_pred_fs_x = torch.autograd.grad(cv_pred_fs, x_fs, grad_outputs=torch.ones_like(cv_pred_fs), retain_graph=True, create_graph=True)[0] 

        g4 = -1*self.k3_0*torch.exp(self.beta_cv*u_pred_fs)
        q4 = -1*self.U_cv*u_pred_fs_x
        
        cv_fs_loss = torch.mean((-self.D_cv*cv_pred_fs_x -g4 + q4*cv_pred_fs)**2)

        # av at f/s conditions
        av_pred_fs = self.AV_net(inputs_fs)
        av_pred_fs_x = torch.autograd.grad(av_pred_fs, x_fs, grad_outputs=torch.ones_like(av_pred_fs), retain_graph=True, create_graph=True)[0] 

        q5 = -1*(self.k4_0*torch.exp(self.alpha_av*u_pred_fs) + self.U_av*u_pred_fs_x)
        
        av_fs_loss = torch.mean((-self.D_av*av_pred_fs_x + q5*av_pred_fs)**2)

        # hole at f/s conditions
        h_pred_fs = self.h_net(inputs_fs)
        h_pred_fs_x = torch.autograd.grad(h_pred_fs, x_fs, grad_outputs=torch.ones_like(h_pred_fs), retain_graph=True, create_graph=True)[0] 

        hole_threshold = 1e-9
        mask = h_pred_fs > hole_threshold
        g6 = torch.zeros_like(h_pred_fs)
        q6 = torch.where(mask, (self.ktp_0 + (self.F*self.D_h)/(self.R*self.T)*u_pred_fs_x), torch.zeros_like(h_pred_fs))

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
        L_pred = self.L_net(t)
        x = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * L_pred 
        
        # Compute PDE residuals
        cd_cv_residual, cd_av_residual, cd_h_residual, poisson_residual = self.pde_residuals(x, t)
        
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
        
        # Get detailed losses
        interior_loss, cv_pde_loss, av_pde_loss, h_pde_loss, poisson_pde_loss = self.interior_loss()
        ic_loss, cv_ic_loss, av_ic_loss, poisson_ic_loss, h_ic_loss, L_ic_loss = self.initial_condition_loss()
        bc_loss, cv_mf_loss, av_mf_loss, u_mf_loss, cv_fs_loss, av_fs_loss, u_fs_loss, h_fs_loss = self.boundary_loss()
        L_physics_loss = self.L_loss()
        
        # Apply weights
        total_loss = (interior_loss + 
                    bc_loss + 
                    ic_loss + 
                    L_physics_loss)
        
        # Create detailed loss dictionary
        loss_dict = {
            'total': total_loss,
            'interior': interior_loss,
            'boundary': bc_loss,
            'initial': ic_loss,
            'L_physics': L_physics_loss,
            # PDE losses
            'cv_pde': cv_pde_loss,
            'av_pde': av_pde_loss,
            'h_pde': h_pde_loss,
            'poisson_pde': poisson_pde_loss,
            # Initial condition losses
            'cv_ic': cv_ic_loss,
            'av_ic': av_ic_loss,
            'poisson_ic': poisson_ic_loss,
            'h_ic': h_ic_loss,
            'L_ic': L_ic_loss,
            # Boundary condition losses
            'cv_mf_bc': cv_mf_loss,
            'av_mf_bc': av_mf_loss,
            'u_mf_bc': u_mf_loss,
            'cv_fs_bc': cv_fs_loss,
            'av_fs_bc': av_fs_loss,
            'u_fs_bc': u_fs_loss,
            'h_fs_bc': h_fs_loss
        }
        
        return loss_dict

    def train_step(self):
        """Perform one training step with detailed loss tracking"""
        self.optimizer.zero_grad()
        loss_dict = self.total_loss()
        loss_dict['total'].backward() 
        self.optimizer.step()
        self.scheduler.step()

        # Convert all losses to float for logging
        loss_dict_float = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
        return loss_dict_float
    
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
        for step in range(self.cfg.training.max_steps):
            loss_dict = self.train_step()
            
            # Store all losses
            for key in loss_history.keys():
                loss_history[key].append(loss_dict[key])
            
            # Print progress with detailed breakdown
            if step % self.cfg.training.rec_results_freq == 0:
                print(f"\n=== Step {step} ===")
                print(f"Total Loss: {loss_dict['total']:.6f}")
                print(f"Interior: {loss_dict['interior']:.6f} | Boundary: {loss_dict['boundary']:.6f} | "
                    f"Initial: {loss_dict['initial']:.6f} | L_Physics: {loss_dict['L_physics']:.6f}")
                
                print("\nPDE Residuals:")
                print(f"  CV PDE: {loss_dict['cv_pde']:.6f} | AV PDE: {loss_dict['av_pde']:.6f}")
                print(f"  Hole PDE: {loss_dict['h_pde']:.6f} | Poisson PDE: {loss_dict['poisson_pde']:.6f}")
                
                print("\nBoundary Conditions:")
                print(f"  m/f interface - CV: {loss_dict['cv_mf_bc']:.6f} | AV: {loss_dict['av_mf_bc']:.6f} | U: {loss_dict['u_mf_bc']:.6f}")
                print(f"  f/s interface - CV: {loss_dict['cv_fs_bc']:.6f} | AV: {loss_dict['av_fs_bc']:.6f} | U: {loss_dict['u_fs_bc']:.6f} | H: {loss_dict['h_fs_bc']:.6f}")
                
                print("\nInitial Conditions:")
                print(f"  CV: {loss_dict['cv_ic']:.6f} | AV: {loss_dict['av_ic']:.6f} | H: {loss_dict['h_ic']:.6f}")
                print(f"  Poisson: {loss_dict['poisson_ic']:.6f} | L: {loss_dict['L_ic']:.6f}")
                
                # Save if specified
                if step % self.cfg.training.save_network_freq == 0 and step > 0:
                    self.save_model(f"outputs/checkpoints_{self.cfg.experiment.name}/model_step_{step}")
                    
                    # Visualize if needed
                    if step % self.cfg.training.rec_inference_freq == 0:
                       self.visualize_predictions(step)
        
        # Final save and print
        final_loss = loss_dict
        print(f"\n=== Final Results (Step {step}) ===")
        print(f"Total Loss: {final_loss['total']:.6f}")
        print("PDE Analysis:")
        print(f"  Worst PDE: {max([('CV', final_loss['cv_pde']), ('AV', final_loss['av_pde']), ('Hole', final_loss['h_pde']), ('Poisson', final_loss['poisson_pde'])], key=lambda x: x[1])}")
        
        self.save_model(f"outputs/checkpoints_{self.cfg.experiment.name}/model_final")
        
        return loss_history
    
    def visualize_predictions(self, step="final"):
        """Simple function to visualize network predictions across input ranges"""
        
        # Create output directory
        plots_dir = f"outputs/plots_{self.cfg.experiment.name}"
        os.makedirs(plots_dir, exist_ok=True)
        
        with torch.no_grad():
            # Define input ranges
            n_spatial = 50
            n_temporal = 50
            
            # Time range (0 to time_scale)
            t_range = torch.linspace(0, self.time_scale, n_temporal).to(self.device)
            
            # Get final film thickness to set spatial range
            t_final = torch.tensor([[float(self.time_scale)]], device=self.device)
            L_final = self.L_net(t_final).item()
            x_range = torch.linspace(0, L_final, n_spatial).to(self.device)
            
            print(f"Plotting predictions over:")
            print(f"  Time range: [0, {self.time_scale:.1f}]")
            print(f"  Spatial range: [0, {L_final:.2e}]")
            
            # Create 2D grid for contour plots
            T_mesh, X_mesh = torch.meshgrid(t_range, x_range, indexing='ij')
            inputs_2d = torch.stack([X_mesh.flatten(), T_mesh.flatten()], dim=1)
            
            # Get 2D predictions
            u_2d = self.potential_net(inputs_2d).reshape(n_temporal, n_spatial)
            cv_2d = self.CV_net(inputs_2d).reshape(n_temporal, n_spatial)
            av_2d = self.AV_net(inputs_2d).reshape(n_temporal, n_spatial)
            h_2d = self.h_net(inputs_2d).reshape(n_temporal, n_spatial)
            
            # Film thickness evolution
            t_1d = t_range.unsqueeze(1)
            L_1d = self.L_net(t_1d).squeeze()
            
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
            axes[0,0].set_title('Potential φ(x,t)')
            plt.colorbar(im1, ax=axes[0,0])
            
            # 2. Cation vacancies
            im2 = axes[0,1].contourf(X_np, T_np, cv_np, levels=20, cmap='Reds')
            axes[0,1].set_xlabel('Position')
            axes[0,1].set_ylabel('Time')
            axes[0,1].set_title('Cation Vacancies c_cv(x,t)')
            plt.colorbar(im2, ax=axes[0,1])
            
            # 3. Anion vacancies
            im3 = axes[0,2].contourf(X_np, T_np, av_np, levels=20, cmap='Blues')
            axes[0,2].set_xlabel('Position')
            axes[0,2].set_ylabel('Time')
            axes[0,2].set_title('Anion Vacancies c_av(x,t)')
            plt.colorbar(im3, ax=axes[0,2])
            
            # 4. Holes
            im4 = axes[1,0].contourf(X_np, T_np, h_np, levels=20, cmap='Purples')
            axes[1,0].set_xlabel('Position')
            axes[1,0].set_ylabel('Time')
            axes[1,0].set_title('Holes c_h(x,t)')
            plt.colorbar(im4, ax=axes[1,0])
            
            # 5. Film thickness
            axes[1,1].plot(L_np, t_np, 'k-', linewidth=3)
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('Film Thickness')
            axes[1,1].set_title('Film Thickness L(t)')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add legends
            axes[1,2].legend(loc='upper left')
            
            plt.suptitle(f'Network Predictions Overview - Step {step}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{plots_dir}/predictions_overview_step_{step}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print statistics
            print(f"\nPrediction Statistics (Step {step}):")
            print("-" * 40)
            print(f"Potential:        {u_np.min():.2e} to {u_np.max():.2e} (mean: {u_np.mean():.2e})")
            print(f"Cation Vacancies: {cv_np.min():.2e} to {cv_np.max():.2e} (mean: {cv_np.mean():.2e})")
            print(f"Anion Vacancies:  {av_np.min():.2e} to {av_np.max():.2e} (mean: {av_np.mean():.2e})")
            print(f"Holes:            {h_np.min():.2e} to {h_np.max():.2e} (mean: {h_np.mean():.2e})")
            print(f"Film Thickness:   {L_np.min():.2e} to {L_np.max():.2e}")
            
            # Check for potential issues"""Simple function to visualize network predictions across input ranges"""
        
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
        """Export combined model for Netron visualization"""
        
        save_path = f"outputs/pinnacle_architecture.onnx"
        os.makedirs("outputs", exist_ok=True)
        
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
                
            def forward(self, x, t):
                # Combine x,t for spatial-temporal networks
                xt_input = torch.cat([x, t], dim=1)
                
                # Get all predictions
                potential = self.potential_network(xt_input)
                cv_conc = self.cation_vacancy_network(xt_input)
                av_conc = self.anion_vacancy_network(xt_input)
                h_conc = self.hole_network(xt_input)
                thickness = self.film_thickness_network(t)
                
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
        
        torch.onnx.export(
            combined,
            (dummy_x, dummy_t),
            save_path,
            input_names=['x_position', 't_time'],
            output_names=['potential', 'cation_vacancy_conc', 'anion_vacancy_conc', 'hole_conc', 'film_thickness']
        )
        
        print(f"✅ Architecture exported!")
        

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = Pinnacle(cfg)
    
    # Train with detailed loss tracking
    os.makedirs(f"outputs/checkpoints_{cfg.experiment.name}",exist_ok=True)
    loss_history = model.train()

    # Export to ONNX after training
    print("\n" + "="*50)
    print("Exporting trained model to ONNX...")
    model.export_for_netron()
    print("="*50)
    
    # Create comprehensive loss plots
    plot_detailed_losses(loss_history,cfg.experiment.name)

def plot_detailed_losses(loss_history,experiment_name):
    """Create comprehensive plots of all loss components"""
    
    # Create output directory
    os.makedirs(f"outputs/plots_{experiment_name}", exist_ok=True)
    
    # Plot 1: Main loss components
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
    
    # Print final analysis
    final_losses = {k: v[-1] for k, v in loss_history.items()}
    print(f"\nFinal Loss Analysis:")
    print(f"Total Loss: {final_losses['total']:.2e}")
    
    pde_losses = {
        'CV PDE': final_losses['cv_pde'],
        'AV PDE': final_losses['av_pde'], 
        'Hole PDE': final_losses['h_pde'],
        'Poisson PDE': final_losses['poisson_pde']
    }
    
    dominant_pde = max(pde_losses, key=pde_losses.get)
    print(f"Dominant PDE: {dominant_pde} ({pde_losses[dominant_pde]:.2e})")
    
    bc_losses = {
        'CV (m/f)': final_losses['cv_mf_bc'],
        'AV (m/f)': final_losses['av_mf_bc'],
        'Potential (m/f)': final_losses['u_mf_bc'],
        'CV (f/s)': final_losses['cv_fs_bc'],
        'AV (f/s)': final_losses['av_fs_bc'],
        'Potential (f/s)': final_losses['u_fs_bc'],
        'Hole (f/s)': final_losses['h_fs_bc']
    }
    
    dominant_bc = max(bc_losses, key=bc_losses.get)
    print(f"Dominant Boundary Condition: {dominant_bc} ({bc_losses[dominant_bc]:.2e})")

if __name__ == "__main__":
    main()

