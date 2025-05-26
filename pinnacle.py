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


class FFN(nn.Module):
    """Fully Connected Feed Forward Neural Net"""

    def __init__(self, cfg, input_dim=2, output_dim=1):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = cfg.nr_layers
        self.layer_size = cfg.layer_size

        self.activation = nn.Tanh()

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

        t = torch.zeros(self.cfg.batch_size.IC, 1, device=self.device)
        x = torch.rand(self.cfg.batch_size.IC, 1, device=self.device) * self.L_initial
        t.requires_grad = True
        inputs = torch.cat([x, t], dim=1)

        #Initial Conditions for film_thickness
        L_initial_pred = self.L_net(t)

        L_initial_loss = torch.mean( (L_initial_pred - self.L_initial)**2 )

        # Cation Vacancy Initial Conditions
        cv_initial_pred = self.CV_net(inputs)

        cv_initial_t = torch.autograd.grad(cv_initial_pred, t, grad_outputs=torch.ones_like(cv_initial_pred), retain_graph=True, create_graph=True)[0]

        cv_initial_loss = torch.mean(cv_initial_pred ** 2) + torch.mean(cv_initial_t ** 2)

        # Anion Vacancy Initial Conditions
        av_initial_pred = self.AV_net(inputs)

        av_initial_t = torch.autograd.grad(av_initial_pred, t, grad_outputs=torch.ones_like(av_initial_pred), retain_graph=True, create_graph=True)[0]

        av_initial_loss = torch.mean(av_initial_pred ** 2) + torch.mean(av_initial_t ** 2)

        # Poission Initial Conditions
        u_initial_pred = self.potential_net(inputs)

        u_inital_t = torch.autograd.grad(u_initial_pred, t, grad_outputs=torch.ones_like(av_initial_pred), retain_graph=True, create_graph=True)[0]

        poisson_initial_loss = torch.mean((u_initial_pred - (self.E_ext - 1e7 * x)) ** 2) + torch.mean(u_inital_t ** 2)

        # Hole Initial Conditions
        h_initial_pred = self.h_net(inputs)

        h_initial_t = torch.autograd.grad(h_initial_pred, t, grad_outputs=torch.ones_like(h_initial_pred), retain_graph=True, create_graph=True)[0]

        h_initial_loss = torch.mean((h_initial_pred - self.c_h0) ** 2) + torch.mean(h_initial_t ** 2)

        total_initial_loss = cv_initial_loss + av_initial_loss + poisson_initial_loss + h_initial_loss + L_initial_loss #Might want to add weighting here

        return total_initial_loss

    def boundary_loss(self):

        k1, k2, k3, k4, k5, ktp, ko2 = self.compute_rate_constants()

        #initialize the time inputs
        t = torch.rand(self.cfg.batch_size.BC, 1, device=self.device,requires_grad=True) * self.time_scale
        
        #m/f interface conditions
        x_mf = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device,requires_grad=True)
        inputs_mf = torch.cat([x_mf,t],dim=1)

        #Predicting the potential at m/f interface on the film side
        u_pred_mf = self.potential_net(inputs_mf)
        u_pred_mf_x = torch.autograd.grad(u_pred_mf, x_mf, grad_outputs=torch.ones_like(u_pred_mf), retain_graph=True, create_graph=True)[0]

        #cv at m/f conditions
        cv_pred_mf = self.CV_net(inputs_mf)
        cv_pred_mf_x = torch.autograd.grad(cv_pred_mf, x_mf, grad_outputs=torch.ones_like(cv_pred_mf), retain_graph=True, create_graph=True)[0] 

        q1 = self.k1_0* torch.exp(self.alpha_cv*(self.E_ext-u_pred_mf)) + self.U_cv*u_pred_mf_x  - (self.Omega*(k2-k5)) # have the option to find d(x,t) via auto diff or by computing w.r.t rate constants, this wll need to be tuned
        cv_mf_loss = torch.mean( (-self.D_cv*cv_pred_mf_x +q1*cv_pred_mf)**2)

        #av at m/f conditions 
        av_pred_mf = self.AV_net(inputs_mf)
        av_pred_mf_x = torch.autograd.grad(av_pred_mf, x_mf, grad_outputs=torch.ones_like(av_pred_mf), retain_graph=True, create_graph=True)[0] 

        g2 = (4/3)*self.k2_0*torch.exp(self.alpha_av*(self.E_ext-u_pred_mf))
        q2 = -1*self.U_av*u_pred_mf_x - (self.Omega*(k2-k5))

        av_mf_loss = torch.mean( (self.D_av*av_pred_mf_x -g2 +q2*av_pred_mf)**2 )

        #potential at m/f conditions
        g3 = self.eps_Ddl* ( (u_pred_mf-self.E_ext)/self.d_Ddl )
        u_mf_loss = torch.mean ((-self.epsilonf*u_pred_mf_x -g3 )**2 )

        #Predict L to compute the location of the f/s interface
        L_pred = self.L_net(t)

        #f/s interface conditions
        x_fs = torch.ones(self.cfg.batch_size.BC,1,device=self.device,requires_grad=True) * L_pred
        inputs_fs = torch.cat([x_fs,t],dim=1)
        
        #Predicting the potential at f/s
        u_pred_fs = self.potential_net(inputs_fs)
        u_pred_fs_x = torch.autograd.grad(u_pred_fs, x_fs, grad_outputs=torch.ones_like(u_pred_fs), retain_graph=True, create_graph=True)[0]

        #cv at f/s conditions
        cv_pred_fs = self.CV_net(inputs_fs)
        cv_pred_fs_x = torch.autograd.grad(cv_pred_fs, x_fs, grad_outputs=torch.ones_like(cv_pred_fs), retain_graph=True, create_graph=True)[0] 

        g4 = -1*self.k3_0*torch.exp(self.beta_cv*u_pred_fs)
        q4 = -1*self.U_cv*u_pred_fs_x
        
        cv_fs_loss = torch.mean((-self.D_cv*cv_pred_fs_x -g4 + q4*cv_pred_fs)**2)

        #av at f/s conditions
        av_pred_fs = self.AV_net(inputs_fs)
        av_pred_fs_x = torch.autograd.grad(av_pred_fs, x_fs, grad_outputs=torch.ones_like(av_pred_fs), retain_graph=True, create_graph=True)[0] 

        q5 = -1*(self.k4_0*torch.exp(self.alpha_av*u_pred_fs) + self.U_av*u_pred_fs_x)
        
        av_fs_loss = torch.mean((-self.D_av*av_pred_fs_x + q5*av_pred_fs)**2)

        #hole at f/s conditions
        h_pred_fs = self.h_net(inputs_fs)
        h_pred_fs_x = torch.autograd.grad(h_pred_fs, x_fs, grad_outputs=torch.ones_like(h_pred_fs), retain_graph=True, create_graph=True)[0] 

        hole_threshold = 1e-9
        mask = h_pred_fs > hole_threshold
        g6 = torch.zeros_like(h_pred_fs)
        q6 = torch.where(mask, (self.ktp_0 + (self.F*self.D_h)/(self.R*self.T)*u_pred_fs_x), torch.zeros_like(h_pred_fs))


        h_fs_loss = torch.mean((-self.D_h*h_pred_fs_x -g6 +q6*h_pred_fs)**2)

        #Potential at f/s loss
        g7 = -1*self.eps_Ddl*0 #assuming that the electric field in bulk solution is 0
        q7 = 0
        r = -self.d_Ddl*(self.eps_film/self.eps_Ddl)*u_pred_fs_x - self.d_dl*(self.eps_film/self.eps_dl)*u_pred_fs_x

        u_fs_loss = torch.mean((u_pred_fs - r)**2)

        total_BC_loss = cv_mf_loss+av_mf_loss+u_mf_loss+cv_fs_loss+av_fs_loss+u_fs_loss+h_fs_loss

        return total_BC_loss

    def interior_loss(self):
        """Compute PDE residuals on interior points"""


        # Sample interior points
        t = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * self.time_scale
        L_pred = self.L_net(t)
        x = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * L_pred 
        
        # Compute PDE residuals
        cd_cv_residual, cd_av_residual, cd_h_residual, poisson_residual = self.pde_residuals(x, t)
        
        pde_loss = (torch.mean(cd_cv_residual**2) + 
                    torch.mean(cd_av_residual**2) + 
                    torch.mean(cd_h_residual**2) + 
                    torch.mean(poisson_residual**2))
    
        return pde_loss
    

    def total_loss(self):
        """Compute total weighted loss"""
        interior_loss = self.interior_loss()
        ic_loss = self.initial_condition_loss()
        bc_loss = self.boundary_loss()
        L_physics_loss = self.L_loss()
        
        # Apply weights
        total_loss = (interior_loss + 
                    bc_loss + 
                    ic_loss + 
                    L_physics_loss)
        
        return total_loss, interior_loss, ic_loss, bc_loss,L_physics_loss

    def train_step(self):
            """Perform one training step."""
            self.optimizer.zero_grad()
            loss,interior_loss,ic_loss,bc_loss,L_physics_loss = self.total_loss()
            loss.backward() 
            self.optimizer.step()
            self.scheduler.step()

            return loss.item(), interior_loss.item(), ic_loss.item(), bc_loss.item(), L_physics_loss.item()
    
    def train(self):
        """Train the model."""
        losses = []
        poisson_losses = []
        cv_losses = []
        av_losses= []
        h_loss = []
        thickness_losses = []
        interior_losses = []
        ic_losses = []
        bc_losses = []
        
        # Training loop
        for step in range(self.cfg.training.max_steps):
            loss, interior_loss, ic_loss, bc_loss,thickness_loss = self.train_step()
            losses.append(loss)
            interior_losses.append(interior_loss)
            ic_losses.append(ic_loss)
            thickness_losses.append(thickness_loss)
            bc_losses.append(bc_loss)
            
            # Print progress
            if step % self.cfg.training.rec_results_freq == 0:
                print(f"Step {step}, Loss: {loss:.6f}, Interior: {interior_loss:.6f}, "
                      f"IC: {ic_loss:.6f}, BC: {bc_loss:.6f}, L_Loss: {thickness_loss:.6f}")
                
                # Save if specified
                if step % self.cfg.training.save_network_freq == 0 and step > 0:
                    self.save_model(f"outputs/checkpoints/model_step_{step}")
                    
        
        # Final save
        print(f"Step {step}, Loss: {loss:.6f}, Interior: {interior_loss:.6f},IC: {ic_loss:.6f}, BC: {bc_loss:.6f}, L_Loss: {thickness_loss:.6f}")
        self.save_model("outputs/checkpoints/model_final")

        
        return losses, interior_losses,ic_losses,bc_losses,thickness_losses
    
    def save_model(self, name):
        """Save model state."""
        torch.save({
            'potential_net': self.potential_net.state_dict(),
            'CV_net': self.CV_net.state_dict(),  # Fix name
            'AV_net': self.AV_net.state_dict(),  # Add missing nets
            'h_net': self.h_net.state_dict(),
            'L_net': self.L_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f"{name}.pt")

@hydra.main(config_path="conf", config_name="config",version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = Pinnacle(cfg)
    
    # Train
    losses, interior_losses,ic_losses,bc_losses,thickness_losses = model.train()
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses, label='Total Loss')
    plt.semilogy(interior_losses, label='Interior Loss')
    plt.semilogy(ic_losses, label='IC Loss')
    plt.semilogy(bc_losses, label='Boundary Loss')
    plt.semilogy(thickness_losses, label='Thickness Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/plots/training_losses.png")
    plt.close()

if __name__ == "__main__":
    main()

