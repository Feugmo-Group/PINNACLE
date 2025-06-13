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

    def compute_gradients(self, x, t, E):
        """Compute the gradients of potential and concentration."""
        x.requires_grad_(True)
        t.requires_grad_(True)

        # Compute Forward pass

        inputs = torch.cat([x, t], dim=1)  # puts all the input values together into one big list
        u_pred = self.potential_net(inputs)  # Acts the neural networks on the inputs to compute u and c
        cv_pred = self.CV_net(inputs)
        av_pred = self.AV_net(inputs)
        h_pred = self.h_net(inputs)

        #These are predicting the non-dimensional values



        # Compute all the time derrivatives
        cv_t = torch.autograd.grad(
            cv_pred, t, grad_outputs=torch.ones_like(cv_pred),
            create_graph=True, retain_graph=True
        )[0]

        #del c_cvhat/ del that

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
    
    

