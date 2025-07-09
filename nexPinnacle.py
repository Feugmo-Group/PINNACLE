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
from hydra.core.hydra_config import HydraConfig
import json

torch.manual_seed(995)#995 is the number stamped onto my necklace
torch.cuda.empty_cache() #Clear gpu cache before every run

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Swoosh(nn.Module):
    def forward(self, x):
        return torch.abs(x) * torch.sigmoid(x)
      
class Swash(nn.Module):
    def forward(self, x):
        return x**2 * torch.sigmoid(x)
      
class SquashSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) + 0.5
      
      
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

        self.ntk_weights = {
        'cv_pde': 1.0,
        'av_pde': 1.0, 
        'h_pde': 1.0,
        'poisson_pde': 1.0,
        'L_physics': 1.0,
        'boundary': 1.0,
        'initial': 1.0
        }


        self.ntk_batch_sizes = {}  # Store computed batch sizes

        self.current_step = 0
        
        #TODO: Add equation of residual 
        self.ntk_loss_registry = {
        'cv_pde': {
            'sampler': self._sample_interior_points,
            'loss_computer': self._compute_cv_pde_residual,
            'batch_size_key': 'interior'
        },
        'av_pde': {
            'sampler': self._sample_interior_points, 
            'loss_computer': self._compute_av_pde_residual,
            'batch_size_key': 'interior'
        },
        'h_pde': {
            'sampler': self._sample_interior_points,
            'loss_computer': self._compute_h_pde_residual, 
            'batch_size_key': 'interior'
        },
        'poisson_pde': {
            'sampler': self._sample_interior_points,
            'loss_computer': self._compute_poisson_residual,
            'batch_size_key': 'interior'
        },
        'L_physics': {
            'sampler': self._sample_L_points,
            'loss_computer': self._compute_L_residual,
            'batch_size_key': 'L'
        },
        'boundary': {
            'sampler': self._sample_boundary_points,
            'loss_computer': self._compute_boundary_residuals_for_ntk,
            'batch_size_key': 'BC'
        },
        'initial': {
            'sampler': self._sample_initial_points,
            'loss_computer': self._compute_ic_residuals_for_ntk,
            'batch_size_key': 'IC'
        }
    }
        
        # BRDR-specific attributes
        self.brdr_weights = {
        'cv_pde': 1.0,
        'av_pde': 1.0, 
        'h_pde': 1.0,
        'poisson_pde': 1.0,
        'L_physics': 1.0,
        'boundary': 1.0,
        'initial': 1.0
    }
    
         # Moving averages for R⁴(t) - track for each loss component
        self.brdr_moving_averages = {
        'cv_pde': 0.0,
        'av_pde': 0.0,
        'h_pde': 0.0, 
        'poisson_pde': 0.0,
        'L_physics': 0.0,
        'boundary': 0.0,
        'initial': 0.0
    }
    
        # Current squared residuals for irdr computation
        self.current_residuals_squared = {}
    
         # Adaptive scale factor
        self.brdr_scale_factor = 1.0
        self.prev_loss = None
        self.prev_grad_norm = None


        # Optimizer
        params = self.parameters()
        self.optimizer = optim.AdamW(
            params,
            lr=cfg.optimizer.adam.lr,
            betas=cfg.optimizer.adam.betas,
            eps=cfg.optimizer.adam.eps,
            weight_decay=cfg.optimizer.adam.weight_decay
        )



        # Scheduler
        if self.cfg.scheduler.type == "RLROP":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=cfg.scheduler.RLROP.factor,
                patience=cfg.scheduler.RLROP.patience,
                threshold=cfg.scheduler.RLROP.threshold,
                min_lr=cfg.scheduler.RLROP.min_lr,
            )
        else: 
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma= self.cfg.scheduler.tf_exponential_lr.decay_rate,
                last_epoch=self.cfg.scheduler.tf_exponential_lr.decay_steps
            )

    #sampling methods to get ntk without boiler plate
    def parameters(self):
        return list(self.potential_net.parameters()) + list(self.CV_net.parameters()) + list(self.AV_net.parameters()) + list(self.h_net.parameters()) + list(self.L_net.parameters())
    
    def _sample_interior_points(self):
        """Generate interior collocation points"""
        batch_size = self.cfg.batch_size.interior
        t = torch.rand(batch_size, 1, device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.interior,1)
        L_pred = self.L_net(torch.cat([t, E], dim=1))
        x = torch.rand(batch_size, 1, device=self.device,requires_grad=True) * L_pred
        return x, t, E

    def _sample_boundary_points(self):
        """Generate boundary collocation points"""
        batch_size = self.cfg.batch_size.BC
        t = torch.rand(batch_size, 1, device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.BC,1)
        # Predict L for f/s boundary
        L_inputs = torch.cat([t, E], dim=1)
        L_pred = self.L_net(L_inputs)
        
        half_batch = batch_size // 2 #Split batch over both boundaries

        # m/f interface points
        x_mf = torch.zeros(half_batch, 1, device=self.device,requires_grad=True)
        t_mf = t[:half_batch]
        E_mf = E[:half_batch]

         # f/s interface points  
        x_fs = L_pred[half_batch:]
        t_fs = t[half_batch:]
        E_fs = E[half_batch:]

        x_boundary = torch.cat([x_mf, x_fs], dim=0)
        t_boundary = torch.cat([t_mf, t_fs], dim=0)
        E_boundary = torch.cat([E_mf, E_fs], dim=0)

        return x_boundary, t_boundary, E_boundary

    def _sample_L_points(self):
        """Generate L physics points"""
        batch_size = self.cfg.batch_size.L
        t = torch.rand(batch_size, 1, device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.L,1)
        return t, E

    def _sample_initial_points(self):
        """Generate initial condition points"""
        batch_size = self.cfg.batch_size.IC
        t = torch.zeros(batch_size, 1, device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.IC,1)
        L_pred = self.L_net(torch.cat([t, E], dim=1))
        x = torch.rand(batch_size, 1, device=self.device,requires_grad=True) * L_pred
        return x, t, E
    
    #May refactor code to use these later.
    def _compute_cv_pde_residual(self, x, t, E):
        """Compute CV PDE residual only"""
        cd_cv_residual, _, _, _ = self.pde_residuals(x, t, E)
        return cd_cv_residual
    
    def _compute_av_pde_residual(self, x, t, E):
        """Compute AV PDE residual only"""
        _, cd_av_residual, _, _ = self.pde_residuals(x, t, E)
        return cd_av_residual

    def _compute_h_pde_residual(self, x, t, E):
        """Compute hole PDE residual only"""
        _, _, cd_h_residual, _ = self.pde_residuals(x, t, E)
        return cd_h_residual

    def _compute_poisson_residual(self, x, t, E):
        """Compute Poisson residual only"""
        _, _, _, poisson_residual = self.pde_residuals(x, t, E)
        return poisson_residual

    def _compute_L_residual(self, t, E):
        """Compute L physics residual"""
        inputs = torch.cat([t, E], dim=1)
        L_pred = self.L_net(inputs)
        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t, E)
        dl_dt_pred = self._grad(L_pred, t)
        dL_dt_physics = (1/self.lc) * self.tc * self.Omega * (k2 - k5)
        return dl_dt_pred - dL_dt_physics
    
    def _compute_boundary_residuals_for_ntk(self,x,t,E):
        """Return only the boundary residuals we need for computing ntk weights"""
        
        batch_size = self.cfg.batch_size.BC
        half_batch = batch_size // 2 #Split batch over both boundaries
        x_mf = x[:half_batch]
        x_fs = x[half_batch:]
        t_mf = t[:half_batch]
        t_fs = t[half_batch:]
        E_mf = E[:half_batch]
        E_fs = E[half_batch:]
        #Predict L and compute derrivative for boundary fluxes
        L_input = torch.cat([t,E],dim=1)
        L_pred = self.L_net(L_input)
        L_pred_t = self._grad(L_pred,t)
        L_pred_t_mf = L_pred_t[:half_batch]  # For m/f interface  


        inputs_mf = torch.cat([x_mf, t_mf,E_mf], dim=1)
        # Predicting the potential at m/f interface
        u_pred_mf = self.potential_net(inputs_mf)
        u_pred_mf_x = self._grad(u_pred_mf,x_mf)

        #cv at m/f conditions
        cv_pred_mf = self.CV_net(inputs_mf)
        cv_pred_mf_x = self._grad(cv_pred_mf,x_mf)
        cv_mf_residual = (-self.D_cv*self.cc/self.lc)*cv_pred_mf_x - self.k1_0*torch.exp(self.alpha_cv*self.phic*(E_mf/self.phic-u_pred_mf)) - (self.U_cv*self.phic/self.lc*u_pred_mf_x-self.lc/self.tc*L_pred_t_mf)*self.cc*cv_pred_mf

        #av at m/f conditions
        av_pred_mf = self.AV_net(inputs_mf)
        av_pred_mf_x = self._grad(av_pred_mf,x_mf)
        av_mf_residual = (-self.D_av*self.cc/self.lc)*av_pred_mf_x - (4/3)*self.k2_0*torch.exp(self.alpha_av*self.phic*(E_mf/self.phic-u_pred_mf)) - (self.U_av*self.phic/self.lc*u_pred_mf_x-self.lc/self.tc*L_pred_t_mf)*av_pred_mf

        #potential at m/f conditions
        u_mf_residual = (self.eps_film*self.phic/self.lc * u_pred_mf_x) - self.eps_Ddl*self.phic*(u_pred_mf-E_mf/self.phic)/self.d_Ddl

        # f/s interface conditions
        inputs_fs = torch.cat([x_fs, t_fs,E_fs], dim=1)

        # Predicting the potential at f/s
        u_pred_fs = self.potential_net(inputs_fs)
        u_pred_fs_x = self._grad(u_pred_fs,x_fs)

        # cv at f/s conditions
        cv_pred_fs = self.CV_net(inputs_fs)
        cv_pred_fs_x = self._grad(cv_pred_fs,x_fs)
        cv_fs_residual = (-self.D_cv*self.cc/self.lc)*cv_pred_fs_x - (self.k3_0*torch.exp(self.beta_cv*self.phic*u_pred_fs) - self.U_cv*self.phic/self.lc*u_pred_fs_x)*cv_pred_fs*self.cc#Check this non-diming here

        #av at f/s conditions
        av_pred_fs = self.AV_net(inputs_fs)
        av_pred_fs_x = self._grad(av_pred_fs,x_fs)
        av_fs_residual = (-self.D_av*self.cc/self.lc)*av_pred_fs_x - (self.k4_0*torch.exp(self.alpha_av*u_pred_fs) - self.U_av*self.phic/self.lc*u_pred_fs_x)*av_pred_fs*self.cc

        #Poisson at f/s conditions
        u_fs_residual = (self.eps_film*self.phic/self.lc * u_pred_fs_x) - (self.eps_Ddl*self.phic*u_pred_fs)

        #Hole at f/s conditions
        h_fs_pred = self.h_net(inputs_fs)
        h_fs_pred_x = self._grad(h_fs_pred,x_fs)

        q = torch.where(h_fs_pred <= (1e-9)/self.c_h0,torch.zeros_like(h_fs_pred),- (self.ktp_0 + (self.F*self.D_h*self.phic)/(self.R*self.T*self.lc)*u_pred_fs_x))
        h_fs_residual = (self.D_h*self.c_h0/self.lc) * h_fs_pred_x - q* self.c_h0*h_fs_pred

        total_residual = torch.cat([cv_mf_residual,cv_fs_residual,av_mf_residual,av_fs_residual,h_fs_residual,u_mf_residual,u_fs_residual])

        return total_residual

    def _compute_ic_residuals_for_ntk(self,x,t,E):
        """Return only the initial residuals we need for computing ntk weights"""
        
        L_input = torch.cat([t,E],dim=1)
        L_initial_pred = self.L_net(L_input)
        inputs = torch.cat([x, t,E], dim=1)

        L_initial_residual = L_initial_pred-self.L_initial/self.lc

        # Cation Vacancy Initial Conditions
        cv_initial_pred = self.CV_net(inputs)
        cv_initial_t = self._grad(cv_initial_pred,t)
        cv_initial_residual = cv_initial_pred + cv_initial_t 

        # Anion Vacancy Initial Conditions
        av_initial_pred = self.AV_net(inputs)
        av_initial_t = self._grad(av_initial_pred,t)
        av_initial_residual = av_initial_pred + av_initial_t 

        # Potential Initial Conditions
        u_initial_pred = self.potential_net(inputs)
        u_initial_t = self._grad(u_initial_pred,t)
        poisson_initial_residual = (u_initial_pred - ((E/self.phic) - (1e7 * (self.lc/self.phic)*x))) + u_initial_t

        # Hole Initial Conditions
        h_initial_pred = self.h_net(inputs)
        h_initial_t = self._grad(h_initial_pred,t)
        h_initial_residual = (h_initial_pred - self.c_h0/self.c_h0) + h_initial_t 

        total_residual = torch.cat([
        L_initial_residual.flatten(),
        cv_initial_residual.flatten(), 
        av_initial_residual.flatten(),
        poisson_initial_residual.flatten(),
        h_initial_residual.flatten()
    ])
    
        return total_residual
    
    def compute_ntk_trace_for_loss(self, loss_name):
        """NTK trace computation for any registered loss"""
        
        loss_config = self.ntk_loss_registry[loss_name]
        
        # Generate collocation points
        sampler = loss_config['sampler']
        points = sampler()
        
        # Compute loss residual
        loss_computer = loss_config['loss_computer']
        residual = loss_computer(*points)
        
        # Determine batch size (one-time calculation)
        if loss_name not in self.ntk_batch_sizes:
            indices = torch.randperm(len(residual),device=self.device)[:256]
            residual_sampled = residual[indices]
            jacobian_sampled = self.compute_jacobian(residual_sampled)
            self.ntk_batch_sizes[loss_name] = self.compute_minimum_batch_size(jacobian_sampled)
            print(f"Computed batch size for {loss_name}: {self.ntk_batch_sizes[loss_name]}")

            trace = self.get_ntk(jacobian_sampled, compute='trace')
            
            return trace, len(jacobian_sampled)

        else:
            # Random sampling
            batch_size = self.ntk_batch_sizes[loss_name]
            indices = torch.randperm(len(residual),device=self.device)[:batch_size]
            residual_sampled = residual[indices]
            jacobian_sampled = self.compute_jacobian(residual_sampled)

            # Compute NTK trace
            trace = self.get_ntk(jacobian_sampled, compute='trace')
            
            return trace, len(jacobian_sampled)
    
    def update_ntk_weights(self):
        """Update all NTK weights"""
        
        traces = {}
        counts = {}
        
        # Compute all traces using the registry
        for loss_name in self.ntk_loss_registry:
            traces[loss_name], counts[loss_name] = self.compute_ntk_trace_for_loss(loss_name)
        
        # Compute weights
        raw_weights = {}
        for loss_name in traces:
            mean_trace_j = traces[loss_name] / counts[loss_name]
            sum_all_mean_traces = sum(traces[i]/counts[i] for i in traces)
            raw_weights[loss_name] = (1.0 / mean_trace_j) * sum_all_mean_traces 
        
        # Normalization
        total_raw_weight = sum(raw_weights.values())
        normalization = len(raw_weights) / total_raw_weight
        
        # Update stored weights
        for loss_name in self.ntk_weights:
            self.ntk_weights[loss_name] = raw_weights[loss_name] * normalization
            
        # Log the weights
        print(f"Updated NTK weights: {self.ntk_weights}")

    def compute_brdr_residuals(self):
        """Compute individual residual terms for BRDR weighting"""
        residuals = {}

        x_int, t_int, E_int = self.sample_interior_points()
        cd_cv_residuals, cd_av_residuals,cd_h_residuals,poisson_residuals = self.pde_residuals(x_int,t_int,E_int)
        pass


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
            x_fs = L_pred
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

        cd_h_residual = (-(self.D_h*self.c_h0/self.lc**2) * h_xx) + (-self.F * self.D_h*self.phic*self.c_h0 * (1 / (self.R * self.T*self.lc**2)) * u_x * h_x) - (self.F * self.D_h * self.phic*self.c_h0 * (1 / (self.R * self.T*self.lc**2)) * h_pred * u_xx)  # Using quasai steady state assumption

        # Poisson Residual Calculation

        poisson_residual = u_xx + (self.F*self.lc**2*self.cc*(1/(self.phic*self.epsilonf)) * (self.z_av * av_pred + self.z_cv * cv_pred))

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
        #print(f"MONOLITH - k2: {k2.mean().item():.6e}, k5: {k5:.6e}, (k2-k5): {(k2-k5).mean().item():.6e}")
        dl_dt_pred = self._grad(L_pred,t)

        dL_dt_physics = (1/self.lc)*self.tc*self.Omega * (k2 - k5)

        return torch.mean((dl_dt_pred-dL_dt_physics)**2)
    

    def initial_condition_loss(self):
        """Compute initial condition losses with individual tracking"""
        t = torch.zeros(self.cfg.batch_size.IC, 1, device=self.device)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.IC,1)

        L_input = torch.cat([t,E],dim=1)
        L_initial_pred = self.L_net(L_input)
        x = torch.rand(self.cfg.batch_size.IC, 1, device=self.device) * L_initial_pred #might change this to l_initial = 1
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
        poisson_initial_loss = torch.mean((u_initial_pred - ((E/self.phic) - (1e7 * (self.lc/self.phic)*x))) ** 2) + torch.mean(u_initial_t ** 2)

        # Hole Initial Conditions
        h_initial_pred = self.h_net(inputs)
        h_initial_t = self._grad(h_initial_pred,t)
        h_initial_loss = torch.mean((h_initial_pred - self.c_h0/self.c_h0) ** 2) + torch.mean(h_initial_t ** 2)


        total_initial_loss = cv_initial_loss + av_initial_loss + poisson_initial_loss + h_initial_loss + L_initial_loss

        return total_initial_loss, cv_initial_loss, av_initial_loss, poisson_initial_loss, h_initial_loss, L_initial_loss


    def boundary_loss(self):
        """Compute boundary losses with individual tracking"""

        t = torch.rand(self.cfg.batch_size.BC,1,device=self.device,requires_grad=True)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.BC,1)
        # m/f interface conditions
        x_mf = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        x_mf.requires_grad_(True)

        #Predict L and compute derrivative for boundary fluxes
        L_input = torch.cat([t,E],dim=1)
        L_pred = self.L_net(L_input)
        L_pred_t = self._grad(L_pred,t)

        inputs_mf = torch.cat([x_mf, t,E], dim=1)
        # Predicting the potential at m/f interface
        u_pred_mf = self.potential_net(inputs_mf)
        u_pred_mf_x = self._grad(u_pred_mf,x_mf)

        #cv at m/f conditions
        cv_pred_mf = self.CV_net(inputs_mf)
        cv_pred_mf_x = self._grad(cv_pred_mf,x_mf)
        cv_mf_residual = (-self.D_cv*self.cc/self.lc)*cv_pred_mf_x - self.k1_0*torch.exp(self.alpha_cv*self.phic*(E/self.phic-u_pred_mf)) - (self.U_cv*self.phic/self.lc*u_pred_mf_x-self.lc/self.tc*L_pred_t)*self.cc*cv_pred_mf
        cv_mf_loss = torch.mean(cv_mf_residual**2)

        #av at m/f conditionsz_
        av_pred_mf = self.AV_net(inputs_mf)
        av_pred_mf_x = self._grad(av_pred_mf,x_mf)
        av_mf_residual = (-self.D_av*self.cc/self.lc)*av_pred_mf_x - (4/3)*self.k2_0*torch.exp(self.alpha_av*self.phic*(E/self.phic-u_pred_mf)) - (self.U_av*self.phic/self.lc*u_pred_mf_x-self.lc/self.tc*L_pred_t)*av_pred_mf
        av_mf_loss = torch.mean(av_mf_residual**2)

        #potential at m/f conditions
        u_mf_residual = (self.eps_film*self.phic/self.lc * u_pred_mf_x) - self.eps_Ddl*self.phic*(u_pred_mf-(E/self.phic))/self.d_Ddl
        u_mf_loss = torch.mean(u_mf_residual**2)

        # f/s interface conditions  
        x_fs = L_pred
        inputs_fs = torch.cat([x_fs, t,E], dim=1)

        # Predicting the potential at f/s
        u_pred_fs = self.potential_net(inputs_fs)
        u_pred_fs_x = self._grad(u_pred_fs,x_fs)

        # cv at f/s conditions
        cv_pred_fs = self.CV_net(inputs_fs)
        cv_pred_fs_x = self._grad(cv_pred_fs,x_fs)
        cv_fs_residual = (-self.D_cv*self.cc/self.lc)*cv_pred_fs_x - (self.k3_0*torch.exp(self.beta_cv*self.phic*u_pred_fs) - self.U_cv*self.phic/self.lc*u_pred_fs_x)*cv_pred_fs*self.cc#Check this non-diming here
        cv_fs_loss = torch.mean(cv_fs_residual**2)

        #av at f/s conditions
        av_pred_fs = self.AV_net(inputs_fs)
        av_pred_fs_x = self._grad(av_pred_fs,x_fs)
        av_fs_residual = (-self.D_av*self.cc/self.lc)*av_pred_fs_x - (self.k4_0*torch.exp(self.alpha_av*u_pred_fs) - self.U_av*self.phic/self.lc*u_pred_fs_x)*av_pred_fs*self.cc
        av_fs_loss = torch.mean(av_fs_residual**2)

        #Poisson at f/s conditions
        u_fs_residual = (self.eps_film*self.phic/self.lc * u_pred_fs_x) - (self.eps_Ddl*self.phic*u_pred_fs)
        u_fs_loss = torch.mean(u_fs_residual**2)

        #Hole at f/s conditions
        h_fs_pred = self.h_net(inputs_fs)
        h_fs_pred_x = self._grad(h_fs_pred,x_fs)

        q = torch.where(h_fs_pred <= (1e-9)/self.c_h0,torch.zeros_like(h_fs_pred),- (self.ktp_0 + (self.F*self.D_h*self.phic)/(self.R*self.T*self.lc)*u_pred_fs_x))
        h_fs_residual = (self.D_h*self.c_h0/self.lc) * h_fs_pred_x - q* self.c_h0*h_fs_pred
        h_fs_loss = torch.mean(h_fs_residual**2)

        total_BC_loss = cv_mf_loss + av_mf_loss + u_mf_loss + cv_fs_loss + av_fs_loss + u_fs_loss + h_fs_loss

        return total_BC_loss, cv_mf_loss, av_mf_loss, u_mf_loss, cv_fs_loss, av_fs_loss, u_fs_loss, h_fs_loss
    
    def interior_loss(self):
        """Compute PDE residuals on interior points with individual tracking"""
    
        # Sample interior points
        t = torch.rand(self.cfg.batch_size.interior, 1, device=self.device)
        single_E = torch.rand(1,1,device=self.device)*(self.cfg.pde.physics.E_max - self.cfg.pde.physics.E_min) + self.cfg.pde.physics.E_min
        E = single_E.expand(self.cfg.batch_size.interior,1)
        L_input = torch.cat([t,E],dim=1)
        L_pred = self.L_net(L_input)

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
    

    def compute_jacobian(self, output):
        """Get the jacobian of output w.r.t network parameters"""
        output = output.reshape(-1)
        grads = torch.autograd.grad(
            output,
            list(self.parameters()),
            (torch.eye(output.shape[0]).to(self.device),),
            is_grads_batched=True, retain_graph=True,allow_unused=True
        )
        valid_grads = [grad.flatten().reshape(len(output), -1) 
                   for grad in grads if grad is not None]
        
        return torch.cat(valid_grads, 1)
    
    def get_ntk(self,jac,compute="trace"):
        """Get the NTK matrix of jac """

        if compute == 'full':
            return torch.einsum('Na,Ma->NM', jac, jac)
        elif compute == 'diag':
            return torch.einsum('Na,Na->N', jac, jac)
        elif compute == 'trace':
            return torch.einsum('Na,Na->', jac, jac)
        else:
            raise ValueError('compute must be one of "full",'
                                + '"diag", or "trace"')
        
    def compute_minimum_batch_size(self, jacobian):
        """Compute minimum batch size for 0.2 approximation error"""
        ntk_diag = self.get_ntk(jacobian, compute='diag')
        
        # Population statistics
        mu_X = torch.mean(ntk_diag)
        sigma_X = torch.std(ntk_diag)
        
        # Handle near-zero mean case
        if mu_X.abs() < 1e-8:
            # Use relative variation instead when mean is tiny
            if sigma_X < 1e-8:
                v_X = 1.0  # Uniform case
            else:
                # Use median as reference instead of mean
                median_X = torch.median(ntk_diag)
                v_X = sigma_X / (median_X.abs() + 1e-8)
        else:
            # Normal coefficient of variation
            v_X = sigma_X / mu_X.abs()
        
        # Clamp to reasonable bounds
        v_X = torch.clamp(v_X, min=0.1, max=5.0)
        
        min_batch_size = int(25 * (v_X ** 2))
        min_batch_size = max(min_batch_size, 32)
        min_batch_size = min(min_batch_size, len(jacobian) // 4)
        
        return min_batch_size
        

    def get_loss_weights(self,param = "ntk"):
        """Get appropriate weights based on weighting strategy"""
        if self.cfg.training.weight_strat == "ntk" and self.current_step >= self.cfg.training.ntk_start_step:
            return self.ntk_weights.copy()
        elif self.cfg.training.weight_strat == "None":
            return {
            'cv_pde': 1,
            'av_pde': 1,
            'h_pde': 1, 
            'poisson_pde': 1,
            'L_physics': 1,
            'boundary': 1,
            'initial': 1,
            }
        else:
            # Regular batch-size based weighting
            return {
                'cv_pde': 1/self.cfg.batch_size.interior,
                'av_pde': 1/self.cfg.batch_size.interior,
                'h_pde': 1/self.cfg.batch_size.interior, 
                'poisson_pde': 1/self.cfg.batch_size.interior,
                'L_physics': 1/self.cfg.batch_size.L,
                'boundary': 1/self.cfg.batch_size.BC,
                'initial': 1/self.cfg.batch_size.IC,
            }
        
    def total_loss(self):
        """Compute total weighted loss with detailed breakdown"""
        
        # Compute individual losses
        interior_loss, cv_pde_loss, av_pde_loss, h_pde_loss, poisson_pde_loss = self.interior_loss()
        ic_loss, cv_ic_loss, av_ic_loss, poisson_ic_loss, h_ic_loss, L_ic_loss = self.initial_condition_loss()
        bc_loss, cv_mf_loss, av_mf_loss, u_mf_loss, cv_fs_loss, av_fs_loss, u_fs_loss, h_fs_loss = self.boundary_loss()
        L_physics_loss = self.L_loss()
        
        # Get appropriate weights based on strategy
        weights = self.get_loss_weights()
        
        # Compute weighted losses consistently
        weighted_losses = {
            'cv_pde': weights['cv_pde'] * cv_pde_loss,
            'av_pde': weights['av_pde'] * av_pde_loss, 
            'h_pde': (1/10000)*weights['h_pde'] * h_pde_loss,
            'poisson_pde': weights['poisson_pde'] * poisson_pde_loss,
            'L_physics': weights['L_physics'] * L_physics_loss,
            'boundary': weights['boundary'] * bc_loss,
            'initial': weights['initial'] * ic_loss,
            
            # Individual components (for logging)
            'cv_ic': weights['initial'] * cv_ic_loss, 
            'av_ic': weights['initial'] * av_ic_loss, 
            'poisson_ic': weights['initial'] * poisson_ic_loss, 
            'h_ic': weights['initial'] * h_ic_loss,
            'L_ic': weights['initial'] * L_ic_loss,
            
            'cv_mf_bc': weights['boundary'] * cv_mf_loss,
            'av_mf_bc': weights['boundary'] * av_mf_loss,
            'u_mf_bc': weights['boundary'] * u_mf_loss,
            'cv_fs_bc': weights['boundary'] * cv_fs_loss,
            'av_fs_bc': weights['boundary'] * av_fs_loss,
            'u_fs_bc': weights['boundary'] * u_fs_loss,
            'h_fs_bc': weights['boundary'] * h_fs_loss,
        }
        
        # Total loss
        total_loss = (weighted_losses['cv_pde'] + weighted_losses['av_pde'] + 
                    weighted_losses['h_pde'] + weighted_losses['poisson_pde'] +
                    weighted_losses['L_physics'] + weighted_losses['boundary'] + 
                    weighted_losses['initial'])
        
        # Add totals for convenience
        weighted_losses.update({
            'total': total_loss,
            'interior': weighted_losses['cv_pde'] + weighted_losses['av_pde'] + 
                    weighted_losses['h_pde'] + weighted_losses['poisson_pde'],
        })
        
        return weighted_losses
    
    def train_step(self):
        """Perform one training step with detailed loss tracking"""
        self.optimizer.zero_grad()
        loss_dict = self.total_loss()
        loss_dict['total'].backward() 

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

    def save_loss_summary(self, final_loss_dict):
        """Save both best loss and final loss"""
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        
        final_total_loss = final_loss_dict['total'].item() if torch.is_tensor(final_loss_dict['total']) else final_loss_dict['total']
        
        # Simple text file with both losses
        with open(os.path.join(hydra_output_dir, "loss_summary.txt"), 'w') as f:
            f.write(f"best_loss={self.best_loss:.6e}\n")
            f.write(f"final_loss={final_total_loss:.6e}\n")
        

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
        
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        checkpoints_dir = os.path.join(hydra_output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)


        print( f"Total Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}" )  
        # Training loop
        for step in tqdm(range(self.cfg.training.max_steps)):
            self.current_step = step

            should_update = (
            self.cfg.training.weight_strat == "ntk" and 
            step >= self.cfg.training.ntk_start_step and 
            step % self.cfg.training.ntk_update_freq == 0
            )

            if should_update:
                self.update_ntk_weights()
            

            loss_dict = self.train_step()
            
            # Store all losses
            for key in loss_history.keys():
                loss_history[key].append(loss_dict[key])
            
            # print progress with detailed breakdown
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
                
                if hasattr(self, '_L_diagnostics'):
                    d = self._L_diagnostics
                    print(f"\nL Physics Diagnostics:")
                    print(f"  k2: {d['k2_mean']:.2e} | k5: {d['k5_mean']:.2e} | (k2-k5): {d['k2_k5_diff']:.2e}")
                    print(f"  Ω(k2-k5): {d['omega_k2_k5']:.2e}")
                    print(f"  dL/dt predicted: {d['dL_dt_pred']:.2e} | dL/dt physics: {d['dL_dt_physics']:.2e}")
                    print(f"  Current L: {d['L_current']:.2e} | log(L): {d['log_L_current']:.2f}")

                current_loss = loss_dict['total']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_checkpoint_path = os.path.join(checkpoints_dir, "model_best")  # Updated path
                    self.save_model(self.best_checkpoint_path)
                    
                # Save if specified
                if step % self.cfg.training.save_network_freq == 0 and step > 0:
                    checkpoint_path = os.path.join(checkpoints_dir, f"model_step_{step}")  # Updated path
                    self.save_model(checkpoint_path)

                    # Visualize if needed
                    if step % self.cfg.training.rec_inference_freq == 0:
                       self.visualize_predictions(step)
        
        # Final save and print
        final_loss = loss_dict
        print(f"\n=== Final Results (Step {step}) ===")
        print(f"Total Loss: {final_loss['total']:.6f}")
        print("PDE Analysis:")
        print(f"  Worst PDE: {max([('CV', final_loss['cv_pde']), ('AV', final_loss['av_pde']), ('Hole', final_loss['h_pde']), ('Poisson', final_loss['poisson_pde'])], key=lambda x: x[1])}")
        

        self.visualize_predictions()
        final_checkpoint_path = os.path.join(checkpoints_dir, "model_final")
        self.save_model(final_checkpoint_path)

        if self.best_checkpoint_path:
            print(f"🔄 Loading best checkpoint for inference...")
            self.load_model(self.best_checkpoint_path)
            print(f"✅ Using best model (loss: {self.best_loss:.6f})")

        final_loss_dict = self.total_loss()
        self.save_loss_summary(final_loss_dict)

        return loss_history
    
    def visualize_predictions(self, step="final"):
        """Visualize network predictions across input ranges - non-dimensional version"""
        
        # Create output directory
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        plots_dir = os.path.join(hydra_output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        with torch.no_grad():
            # Define input ranges (all dimensionless)
            n_spatial = 50
            n_temporal = 50
            
            # Fix a representative dimensionless potential for visualization
            E_hat_fixed = torch.tensor([[0.8/self.phic]], device=self.device)  # Normalized E
            
            # Dimensionless time range (0 to 1)
            t_hat_range = torch.linspace(0, 1.0, n_temporal).to(self.device)
            
            # Get final dimensionless film thickness to set spatial range
            t_hat_final = torch.tensor([[1.0]], device=self.device)
            L_hat_final = self.L_net(torch.cat([t_hat_final, E_hat_fixed],dim=1)).item()
            x_hat_range = torch.linspace(0, L_hat_final, n_spatial).to(self.device)
            
            print(f"Plotting predictions over:")
            print(f"  Dimensionless time range: [0, 1.0]")
            print(f"  Dimensionless spatial range: [0, {L_hat_final:.2f}]")
            print(f"  Fixed dimensionless potential: {E_hat_fixed.item():.3f}")
            
            # Create 2D grid for contour plots
            T_hat_mesh, X_hat_mesh = torch.meshgrid(t_hat_range, x_hat_range, indexing='ij')
            E_hat_mesh = torch.full_like(T_hat_mesh, E_hat_fixed.item())
            
            # Stack inputs for 3D networks
            inputs_3d = torch.stack([
                X_hat_mesh.flatten(), 
                T_hat_mesh.flatten(), 
                E_hat_mesh.flatten()
            ], dim=1)
            
            # Get 3D network predictions (all dimensionless)
            phi_hat_2d = self.potential_net(inputs_3d).reshape(n_temporal, n_spatial)
            c_cv_hat_2d = self.CV_net(inputs_3d).reshape(n_temporal, n_spatial)
            c_av_hat_2d = self.AV_net(inputs_3d).reshape(n_temporal, n_spatial)
            c_h_hat_2d = self.h_net(inputs_3d).reshape(n_temporal, n_spatial)
            
            # Film thickness evolution (dimensionless)
            t_hat_1d = t_hat_range.unsqueeze(1)
            E_hat_1d = torch.full_like(t_hat_1d, E_hat_fixed.item())
            L_inputs_1d = torch.cat([t_hat_1d, E_hat_1d], dim=1)
            L_hat_1d = self.L_net(L_inputs_1d).squeeze() 
            
            # Convert to numpy
            t_hat_np = t_hat_range.cpu().numpy()
            x_hat_np = x_hat_range.cpu().numpy()
            T_hat_np, X_hat_np = np.meshgrid(t_hat_np, x_hat_np, indexing='ij')
            
            phi_hat_np = phi_hat_2d.cpu().numpy()
            c_cv_hat_np = c_cv_hat_2d.cpu().numpy()
            c_av_hat_np = c_av_hat_2d.cpu().numpy()
            c_h_hat_np = c_h_hat_2d.cpu().numpy()
            L_hat_np = L_hat_1d.cpu().numpy()
            
            # Create plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Dimensionless potential field
            im1 = axes[0,0].contourf(X_hat_np, T_hat_np, phi_hat_np, levels=50, cmap='RdBu_r')
            axes[0,0].set_xlabel('Dimensionless Position x̂')
            axes[0,0].set_ylabel('Dimensionless Time t̂')
            axes[0,0].set_title(f'Dimensionless Potential φ̂(x̂,t̂) at Ê={E_hat_fixed.item():.3f}')
            plt.colorbar(im1, ax=axes[0,0])
            
            # 2. Dimensionless cation vacancies
            im2 = axes[0,1].contourf(X_hat_np, T_hat_np, c_cv_hat_np, levels=50, cmap='Reds')
            axes[0,1].set_xlabel('Dimensionless Position x̂')
            axes[0,1].set_ylabel('Dimensionless Time t̂')
            axes[0,1].set_title(f'Dimensionless Cation Vacancies ĉ_cv at Ê={E_hat_fixed.item():.3f}')
            plt.colorbar(im2, ax=axes[0,1])
            
            # 3. Dimensionless anion vacancies
            im3 = axes[0,2].contourf(X_hat_np, T_hat_np, c_av_hat_np, levels=50, cmap='Blues')
            axes[0,2].set_xlabel('Dimensionless Position x̂')
            axes[0,2].set_ylabel('Dimensionless Time t̂')
            axes[0,2].set_title(f'Dimensionless Anion Vacancies ĉ_av at Ê={E_hat_fixed.item():.3f}')
            plt.colorbar(im3, ax=axes[0,2])
            
            # 4. Dimensionless holes
            im4 = axes[1,0].contourf(X_hat_np, T_hat_np, c_h_hat_np, levels=50, cmap='Purples')
            axes[1,0].set_xlabel('Dimensionless Position x̂')
            axes[1,0].set_ylabel('Dimensionless Time t̂')
            axes[1,0].set_title(f'Dimensionless Holes ĉ_h at Ê={E_hat_fixed.item():.3f}')
            plt.colorbar(im4, ax=axes[1,0])
            
            # 5. Dimensionless film thickness
            axes[1,1].plot(t_hat_np, L_hat_np, 'k-', linewidth=3)
            axes[1,1].set_xlabel('Dimensionless Time t̂')
            axes[1,1].set_ylabel('Dimensionless Film Thickness L̂')
            axes[1,1].set_title(f'Dimensionless Film Thickness L̂(t̂) at Ê={E_hat_fixed.item():.3f}')
            axes[1,1].grid(True, alpha=0.3)
            
            # 6. Potential profile vs spatial position at fixed time
            x_hat_sweep = torch.linspace(0, L_hat_final, 50).to(self.device)
            t_hat_mid = torch.full((50, 1), 0.5, device=self.device)  # Middle time
            E_hat_mid = torch.full((50, 1), E_hat_fixed.item(), device=self.device)
            
            x_sweep_inputs = torch.cat([x_hat_sweep.unsqueeze(1), t_hat_mid, E_hat_mid], dim=1)
            phi_vs_x = self.potential_net(x_sweep_inputs).cpu().numpy()
            
            axes[1,2].plot(x_hat_sweep.cpu().numpy(), phi_vs_x, 'r-', linewidth=2)
            axes[1,2].set_xlabel('Dimensionless Position x̂')
            axes[1,2].set_ylabel('Dimensionless Potential φ̂')
            axes[1,2].set_title(f'Potential Profile φ̂(x̂) at t̂=0.5, Ê={E_hat_fixed.item():.3f}')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Dimensionless Network Predictions Overview - Step {step}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, f"predictions_overview_step_{step}.png")
            plt.savefig(plot_path, dpi=500, bbox_inches='tight')
            plt.close()
            
            # Print dimensionless statistics
            print(f"\nDimensionless Prediction Statistics (Step {step}) at Ê={E_hat_fixed.item():.3f}:")
            print("-" * 60)
            print(f"Potential φ̂:          {phi_hat_np.min():.3f} to {phi_hat_np.max():.3f} (mean: {phi_hat_np.mean():.3f})")
            print(f"Cation Vacancies ĉ_cv: {c_cv_hat_np.min():.3f} to {c_cv_hat_np.max():.3f} (mean: {c_cv_hat_np.mean():.3f})")
            print(f"Anion Vacancies ĉ_av:  {c_av_hat_np.min():.3f} to {c_av_hat_np.max():.3f} (mean: {c_av_hat_np.mean():.3f})")
            print(f"Holes ĉ_h:             {c_h_hat_np.min():.3f} to {c_h_hat_np.max():.3f} (mean: {c_h_hat_np.mean():.3f})")
            print(f"Film Thickness L̂:      {L_hat_np.min():.3f} to {L_hat_np.max():.3f}")
            
            # Convert back to dimensional units for reference
            print(f"\nCorresponding Dimensional Values:")
            print(f"Time scale: {self.tc:.1e} s")
            print(f"Length scale: {self.lc:.1e} m")
            print(f"Potential scale: {self.phic:.3f} V")
            print(f"Final dimensional thickness: {L_hat_np.max() * self.lc * 1e9:.2f} nm")



    def generate_polarization_curve(self, n_points=50):
        """Generate polarization curve at specified time  hh   """
        
        # Create output directory
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        plots_dir = os.path.join(hydra_output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        t_hat_eval = 1.0  # Use final dimensionless time by default
        
        print(f"Generating polarization curve at t̂={t_hat_eval}")
        
        # Define dimensionless potential range
        E_hat_min = self.cfg.pde.physics.E_min / self.phic
        E_hat_max = self.cfg.pde.physics.E_max / self.phic
        
        
        with torch.no_grad():
            # Create dimensionless potential sweep
            E_hat_values = torch.linspace(E_hat_min, E_hat_max, n_points, device=self.device)
            j={'total':[],'j1':[],'j2':[],'j3':[],'jtp':[]}
            
            for E_hat_val in E_hat_values:
                # Use dimensionless quantities throughout
                t_hat_tensor = torch.tensor([[t_hat_eval]], device=self.device)
                E_hat_tensor = torch.tensor([[E_hat_val.item()]], device=self.device)

                # Get dimensionless film thickness
                L_hat_val = self.L_net(torch.cat([t_hat_tensor, E_hat_tensor], dim=1))

                # Evaluate at interfaces
                x_hat_fs = L_hat_val  # f/s interface
                x_hat_mf = torch.zeros_like(L_hat_val)  # m/f interface

                inputs_fs = torch.cat([x_hat_fs, t_hat_tensor, E_hat_tensor], dim=1)
                inputs_mf = torch.cat([x_hat_mf, t_hat_tensor, E_hat_tensor], dim=1)

                # Get dimensionless concentrations and rate constants
                k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants(t_hat_tensor, E_hat_tensor,
                                                                                single=True)

                h_hat_fs = self.h_net(inputs_fs)  # Dimensionless hole concentration
                cv_hat_mf = self.CV_net(inputs_mf)  # Dimensionless CV concentration
                av_hat_fs = self.AV_net(inputs_fs)

                #Compute Currents
                j1 = (8.0/3.0)*self.F*k1*cv_hat_mf*self.cc
                j2 = (8.0/3.0)*self.F*k2
                j3 = (1.0/3.0)*self.F*k3*av_hat_fs*self.cc
                jtp = self.F*ktp*(self.c_H**9)*h_hat_fs*self.chc
                j_total = j1 +j2 + j3 + jtp
                j["total"].append(j_total.item())
                j["j1"].append(j1.item())
                j["j2"].append(j2.item())
                j["j3"].append(j3.item())
                j["jtp"].append(jtp.item())

            # Convert to numpy for plotting
            E_np = E_hat_values.cpu().numpy()*self.phic
            j_np = np.array(j["total"])
            j_1_np = np.array(j['j1'])
            j_2_np = np.array(j['j2'])
            j_3_np = np.array(j['j3'])
            j_tp_np = np.array(j['jtp'])

            # Create polarization curve plot
            plt.figure(figsize=(10, 6))
            plt.plot(E_np, j_np, 'b-', linewidth=2, label='Total Current')
            plt.plot(E_np, j_1_np, 'r-', linewidth=2, label='Current due to R1')
            plt.plot(E_np, j_2_np, 'g-', linewidth=2, label='Current due to R2')
            plt.plot(E_np, j_3_np, 'o-', linewidth=2, label='Current due to R3')
            plt.plot(E_np, j_tp_np, 'm-', linewidth=2, label='Current due to RTP')
            plt.xlabel('Applied Potential E')
            plt.ylabel('Current Density j')
            plt.title(f'Polarization Curve at t̂={t_hat_eval}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.legend()
            

            # Save plot
            plot_path = os.path.join(plots_dir, f"polarization_curve_dimensionless.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

def plot_detailed_losses(loss_history):
    """Create comprehensive plots of all loss components"""

    # Create output directory
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    plots_dir = os.path.join(hydra_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

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
    plot_path = os.path.join(plots_dir, f"losses.png")
    plt.savefig(plot_path, dpi=500, bbox_inches='tight')
    plt.close()
    

@hydra.main(config_path="conf/", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = Nexpinnacle(cfg)
    
    # Train with detailed loss tracking
    loss_history = model.train()
    
    #Plot out the loss curve
    plot_detailed_losses(loss_history)

    model.generate_polarization_curve()

if __name__ == "__main__":
    main()

