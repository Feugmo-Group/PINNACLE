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
    def __init__(self,cfg,input_dim=2,output_dim=1):
        super(FFN,self).__init__()
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
    def __init__(self,cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks eaach having inputs x,t and outputs their name sake
        self.potential_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)
        #Need to create a seperate net for the concenration of every species of interest
        self.CV_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device) #Cation Vacany
        self.AV_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device) #Anion Vacancy
        self.h_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device) #Hole
        

        # Physics parameters
        self.D_cv = cfg.pde.physics.D_cv
        self.D_av = cfg.pde.physics.D_av
        self.D_h = cfg.pde.physics.D_h
        self.z_cv = cfg.pde.physics.z_cv
        self.z_av = cfg.pde.physics.z_av
        self.z_h = cfg.pde.physics.z_h
        self.F = cfg.pde.physics.F
        self.R = cfg.pde.physics.R
        self.T = cfg.pde.physics.T 
        self.epsilon = cfg.pde.physics.epsilon
        self.epsilonr = cfg.pde.physics.epsilonr
        self.k1 = cfg.pde.rates.k1
        self.k2 = cfg.pde.rates.k2
        self.k3 = cfg.pde.rates.k3
        self.k4 = cfg.pde.rates.k4
        self.k5 = cfg.pde.rates.k5
        self.ktp = cfg.pde.rates.ktp
        self.ko2 = cfg.pde.rates.o2
        self.alpha1 = cfg.pde.rates.o2
        self.alpha2 = cfg.pde.rates.o2
        self.alpha3= cfg.pde.rates.o2
        self.alpha4 = cfg.pde.rates.o2
        self.alpha5 = cfg.pde.rates.o2
        self.alphatp = cfg.pde.rates.o2
        self.alphao2 = cfg.pde.rates.o2
        self.delta3 = cfg.pde.chemistry.delta3
        self.time_scale = cfg.domain.time.time_scale
        self.L_initial = cfg.domain.initial.L_initial
        


        # Optimizer
        params = list(self.potential_net.parameters()) + list(self.CV_net.parameters()) + list(self.AV_net.parameters()) + list(self.h_net.parameters())
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
        self.poisson_weight = cfg.pnp.weights.poisson_weight
        self.nernst_weight = cfg.pnp.weights.nernst_weight
        self.bc_weight = cfg.pnp.weights.bc_weight

    def compute_gradients(self, x, t):
        """Compute the gradients of potential and concentration."""
        x.requires_grad_(True)
        t.requires_grad_(True)

        # Compute Forward pass
        inputs = torch.cat([x, t], dim=1) #puts all the input values together into one big list
        u_pred = self.potential_net(inputs) #Acts the neural networks on the inputs to compute u and c 
        cv_pred = self.CV_net(inputs)
        av_pred = self.AV_net(inputs)
        h_pred = self.h_net(inputs)


        #Compute all the time derrivatives
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

        #Compute all the derrivatives we need w.r.t x (c derrivatives w.r.t calculated once here and then second calculated when we calculate the flux)
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

    
        #Second Derrivatives
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
        

        return u_pred,cv_pred,av_pred,h_pred,cv_t,av_t,h_t,u_x,cv_x,av_x,h_x,u_xx,cv_xx,av_xx,h_xx
    
    #enforce alll the pde losses
    def pde_residuals(self,x,t):
        """Compute the residuals due to every PDE"""
        u_pred,cv_pred,av_pred,e_pred,h_pred,cv_t,av_t,e_t,h_t,u_x,cv_x,av_x,e_x,h_x,u_xx,cv_xx,av_xx,h_xx = self.compute_gradients()

        #Convection-Diffusion Formulation of Nersnt-Planck 
        cd_cv_residual = cv_t + (-self.D_cv*cv_xx) + (-self.U_cv*u_x*cv_x) - (self.U_cv*cv_pred*u_xx)

        cd_av_residual = av_t + (-self.D_av*av_xx) + (-self.U_av*u_x*av_x) - (self.U_av*av_pred*u_xx)

        cd_h_residual = h_t + (-self.D_h*cv_xx) + (-self.F*self.D_h*(1/self.R*self.T)*u_x*h_x) - (self.F*self.D_h*(1/self.R*self.T)*h_pred*u_xx) #Different from ion convection-diffusion, we are ignoring recombination terms as a simpllifying assumtpion

        #Poisson Residual Calculation

        poisson_residual = -self.epsilonf*u_xx - (self.F*(self.z_av*av_pred+self.z_cv*cv_pred))

        return cd_cv_residual,cd_av_residual,cd_h_residual,poisson_residual

    def get_L(self, t):
        pass
    
    def compute_rate_constants(self):

        #predict the potential on the m/f (x=0) boundary

        x_mf = torch.zeros(self.cfg.batch_size.rate, 1, device=self.device)
        t_mf = torch.rand(self.cfg.batch_size.rate,1,device=self.device) * self.time_scale

        inputs_mf = torch.cat([x_mf,t_mf],dim=1)
        u_mf = self.potential_net(inputs_mf)
         
        #k1 computation
        k1 = self.k1 * np.exp(self.alpha_1*3*self.F*self.R*self.T*u_mf)

        #k2 computation
        k2 = self.k2 * np.exp(self.alpha_2*2*self.F*1/(self.R*self.T)*u_mf)

        #predict the potential on the f/s(x=L) boundary

        x_fs = torch.ones_like(self.cfg.batch_size.rate,1,device=self.device) * self.L_initial #Need to figure out this whole recursive L thing
        t_fs = torch.rand(self.cfg.batch_size.rate,1,device=self.device) * self.time_scale #Might be redundant

        inputs_fs = torch.cat([x_fs,t_fs],dim=1)
        u_fs = self.potential_net(inputs_fs)

        #k3 computation
        k3 = self.k3 * np.exp(self.alhpa3*(3-self.delta)*self.F*1/(self.R*self.T)*u_fs)

        #k4 computation
        k4 = self.k4

        #k5 compuation
        k5 = self.k5 * 1 # FIX THIS RELATED TO QUESTION OF HYDROGEN CONC

        #compute the concentration of holes at the f/s interface
        c_h_fs = self.h_net(inputs_fs)
    
        #ktp computation
        ktp = self.ktp*c_h_fs*np.exp(self.alphatp*self.F*1/(self.R*self.T)*u_fs)

        #ko2 computation

        ko2 = self.ko2*np.exp(self.alphao2*2*self.F*1/(self.R*self.T)*(self.phi_ext-self.phi_o2)) #Whatever the fuck phi_o2 even is
                              
        return k1, k2, k3, k4, k5, ktp, ko2
    

    def initial_condition_loss(self):
        pass

    def boundary_loss(self):

        k1,k2,k3,k4,k5,ktp,ko2 = self.compute_rate_constants()

        t = torch.rand(self.cfg.batch_size.BC,1,device=self.device) * self.time_scale

        #m/f Boundary Conditions fllux_cv = k1*cv flux_ov = k2 
        x_mf = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        inputs_mf = torch.cat([x_mf,t],dim=1)
        u_mf_pred = self.potential_net(inputs_mf)
        u_mf_x = torch.autograd.grad(u_mf_pred,x_mf,grad_outputs=torch.ones_like(u_mf_pred),retain_graph=True,create_graph=True)[0]
        cv_mf_pred = self.CV_net(inputs_mf)
        cv_mf_x = torch.autograd.grad(cv_mf_pred,x_mf,grad_outputs=torch.ones_like(cv_mf_pred),retain_graph=True,create_graph=True)[0]
        flux_cv_mf = -self.D_cv(cv_mf_x + ((self.z_cv*self.F*self.D_cv)/(self.R*self.T))*cv_mf_pred*u_mf_x)
        av_mf_pred = self.AV_net(inputs_mf)
        av_mf_x = torch.autograd.grad(av_mf_pred,x_mf,grad_outputs=torch.ones_like(av_mf_pred),retain_graph=True,create_graph=True)[0]
        flux_av_mf = -self.D_av(av_mf_x + ((self.z_av*self.F*self.D_av)/(self.R*self.T))*av_mf_pred*u_mf_x)

        mf_bc_loss = torch.mean((flux_cv_mf-k1*cv_mf_pred)**2) + torch.mean((flux_av_mf - k2)**2)


        #f/s Boundary Conditions fllux_cv = -k3 flux_ov = k4*c_ov 
        x_fs = torch.ones(self.cfg.batch_size.BC, 1, device=self.device) * self.L_initial #again will have to deal with moving L here
        inputs_fs = torch.cat([x_fs,t],dim=1)
        u_fs_pred = self.potential_net(inputs_fs)
        u_fs_x = torch.autograd.grad(u_fs_pred,x_fs,grad_outputs=torch.ones_like(u_fs_pred),retain_graph=True,create_graph=True)[0]
        cv_fs_pred = self.CV_net(inputs_fs)
        cv_fs_x = torch.autograd.grad(cv_fs_pred,x_fs,grad_outputs=torch.ones_like(cv_fs_pred),retain_graph=True,create_graph=True)[0]
        flux_cv_fs = -self.D_cv(cv_fs_x + ((self.z_cv*self.F*self.D_cv)/(self.R*self.T))*cv_fs_pred*u_fs_x)
        av_fs_pred = self.AV_net(inputs_fs)
        av_fs_x = torch.autograd.grad(av_fs_pred,x_fs,grad_outputs=torch.ones_like(av_fs_pred),retain_graph=True,create_graph=True)[0]
        flux_av_fs = -self.D_av(av_fs_x + ((self.z_av*self.F*self.D_av)/(self.R*self.T))*av_fs_pred*u_fs_x)

        fs_bc_loss = torch.mean( (flux_cv_fs + k3)**2 ) + torch.mean( (flux_cv_fs + k3)**2 )   

        

