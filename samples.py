"""Sample problems to evaluate the performance of the ode integration and 
adjoint backward pass routines.
"""

import torch

from pyoptmat import models, flowrules, hardening
from pyoptmat.temperature import ConstantParameter as CP

class Chaboche(torch.nn.Module):
    """
    The Chaboche viscoplastic constitutive model, as defined in

    Args:
        nbackstress (int): number of backstresses, system size is 3 + this number
    """
    def __init__(self, nbackstress, t_max = 10.0, e_range = [0.1, 1.0], period = 1.0):
        super().__init__()

        self.nbackstress = nbackstress
        self.size = 2 + self.nbackstress

        self.E = torch.nn.Parameter(torch.tensor(10.0))

        self.n = torch.nn.Parameter(torch.tensor(5.0))
        self.eta = torch.nn.Parameter(torch.tensor(2.0))

        self.s0 = torch.nn.Parameter(torch.tensor(1.0))
        
        self.R = torch.nn.Parameter(torch.tensor(10.0))
        self.d = torch.nn.Parameter(torch.tensor(1.0))

        self.C = torch.nn.Parameter(torch.linspace(0.1,1,self.nbackstress))
        self.g = torch.nn.Parameter(torch.linspace(0.1,0.5, self.nbackstress))

        self.isotropic = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))
        self.kinematic = hardening.ChabocheHardeningModel(CP(self.C), CP(self.g))
        self.flowrule = flowrules.IsoKinViscoplasticity(CP(self.n), CP(self.eta), CP(self.s0), 
                self.isotropic, self.kinematic)
        self.model = models.InelasticModel(CP(self.E), self.flowrule)

        self.t_max = t_max
        self.e_range = e_range
        self.period = period

    def setup(self, nbatch, ntime):
        """Setup and return time values and initial conditions for a given batch

        Args:
            nbatch (int): batch size
            ntime (int): number of time steps
        """
        time = torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        y0 = torch.zeros(nbatch, self.size)

        return time, y0

    def force(self, t):
        """Return the driving force

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
        """
        return torch.linspace(*self.e_range, t.shape[-1], device = t.device) * torch.sin(
                2.0 * torch.pi / self.period * t) 

    def rate(self, t, y, force):
        """
        Return the state rate

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
        """
        rate, _, _, _ = self.model(t, y, force, torch.zeros_like(force))

        return rate

    def jacobian(self, t, y, force):
        """
        Return the problem jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """
        _, jac, _, _ = self.model(t, y, force, torch.zeros_like(force))

        return jac

class Neuron(torch.nn.Module):
    """
    ODE model of coupled Neurons, given in

    Schwemmer and Lewis. "The theory of weakly coupled oscillators" 2012

    Args:
        nneurons (int): number of coupled neurons, actual system size
            is four times this

    Keyword Args:
        C_range (float,float): range of values of C parameter
        g_Na_range (float,float): range of values of g_Na parameter
        E_Na_range (float,float): range of values of E_Na parameter
        g_K_range (float,float): range of values of g_K parameter
        E_K_range (float,float): range of values of E_K parameter
        g_L_range (float,float): range of values of g_L parameter
        E_L_range (float,float): range of values of E_L parameter
        m_inf_range (float,float): range of values of m_inf parameter
        tau_m_range (float,float): range of values of tau_m parameter
        h_inf_range (float,float): range of values of h_inf parameter
        tau_h_range (float,float): range of values of tau_h parameter
        n_inf_range (float,float): range of values of n_inf parameter
        tau_n_range (float,float): range of values of tau_n parameter
        g_C_range (float,float): range of values of g_C parameter
        N_cycles (int): number of cycles to try for
        I_max_range ((float,float)): max current range
        I_period_range ((float,float)): current cycle period range


    """
    def __init__(self, nneurons, 
            C_range = [0.1,1.0], g_Na_range = [0.1,1.0], E_Na_range = [0.1,1.0], 
            g_K_range = [0.1,1.0], E_K_range = [0.1,1.0], g_L_range = [0.1,1.0], 
            E_L_range = [0.1,1.0], m_inf_range = [0.1,1.0],
            tau_m_range = [0.5,5.0], h_inf_range = [0.1,1.0], tau_h_range = [1.5,15.0],
            n_inf_range = [0.1,1.0], tau_n_range = [1.0,10.0], g_C_range = [0.001,0.01],
            t_max = 10.0, I_max_range = [0.1, 1.0], I_period_range = [0.5,2]):
        super().__init__()

        self.nneurons = nneurons
        self.size = 4 * self.nneurons
        
        self.C = torch.nn.Parameter(torch.linspace(*C_range, self.nneurons))
        self.g_Na = torch.nn.Parameter(torch.linspace(*g_Na_range,self.nneurons))
        self.E_Na = torch.nn.Parameter(torch.linspace(*E_Na_range, self.nneurons))
        self.g_K = torch.nn.Parameter(torch.linspace(*g_K_range, self.nneurons))
        self.E_K = torch.nn.Parameter(torch.linspace(*E_K_range, self.nneurons))
        self.g_L = torch.nn.Parameter(torch.linspace(*g_L_range, self.nneurons))
        self.E_L = torch.nn.Parameter(torch.linspace(*E_L_range, self.nneurons))
        self.m_inf = torch.nn.Parameter(torch.linspace(*m_inf_range, self.nneurons))
        self.tau_m = torch.nn.Parameter(torch.linspace(*tau_m_range, self.nneurons))
        self.h_inf = torch.nn.Parameter(torch.linspace(*h_inf_range, self.nneurons))
        self.tau_h = torch.nn.Parameter(torch.linspace(*tau_h_range, self.nneurons))
        self.n_inf = torch.nn.Parameter(torch.linspace(*n_inf_range, self.nneurons))
        self.tau_n = torch.nn.Parameter(torch.linspace(*tau_n_range, self.nneurons))
        self.g_C = torch.nn.Parameter(torch.linspace(*g_C_range, self.nneurons))

        self.t_max = t_max
        self.I_max_range = I_max_range
        self.I_period_range = I_period_range

    def setup(self, nbatch, ntime):
        """Setup and return time values and initial conditions for a given batch

        Args:
            nbatch (int): batch size
            ntime (int): number of time steps
        """
        time = torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        y0 = torch.zeros(nbatch, self.size)

        return time, y0

    def force(self, t):
        """Return the driving force

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
        """
        return torch.linspace(*self.I_max_range, t.shape[-1], device = t.device) * torch.sin(
                2.0 * torch.pi / torch.linspace(*self.I_period_range, t.shape[-1],
                    device = t.device) * t) 

    def rate(self, t, y, force):
        """
        Return the state rate

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
        """
        V = y[...,0::4]
        m = y[...,1::4]
        h = y[...,2::4]
        n = y[...,3::4]

        ydot = torch.zeros_like(y)
        
        # V
        ydot[...,0::4] = 1.0 / self.C[None,...] * (
                -self.g_Na[None,...] * m**3.0 * h * (V - self.E_Na[None,...])
                -self.g_K[None,...] * n**4.0 * (V - self.E_K[None,...])
                -self.g_L[None,...] * (V - self.E_L[None,...]) + force[...,None])
        # Coupling term
        dV = torch.sum(self.g_C[None,...]*(V[...,:,None] - V[...,None,:]) / self.C[None,...], dim = -1)
        ydot[...,0::4] += dV

        # m
        ydot[...,1::4] = (self.m_inf - m) / self.tau_m

        # h 
        ydot[...,2::4] = (self.h_inf - h) / self.tau_h

        # n
        ydot[...,3::4] = (self.n_inf - n) / self.tau_n

        return ydot

    def jacobian(self, t, y, force):
        """
        Return the problem jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """
        J = torch.zeros(y.shape + y.shape[-1:], device = y.device)

        V = y[...,0::4]
        m = y[...,1::4]
        h = y[...,2::4]
        n = y[...,3::4]

        # V, V
        J[...,0::4,0::4] = torch.diag_embed(1.0 / self.C[None,...] * (
                -self.g_L[None,...] 
                -self.g_Na[None,...]*h*m**3.0
                -self.g_K[None,...]*n**4.0))
        # Coupling term
        J[...,0::4,0::4] -= self.g_C[None,...] / self.C[None,...] 
        
        J[...,0::4,0::4] += torch.eye(self.nneurons, device = y.device).expand(
                *y.shape[:-1], -1, -1) * torch.sum(self.g_C / self.C)
        
        # V, m
        J[...,0::4,1::4] = torch.diag_embed(-1.0 / self.C[None,...] * (
                3.0 * self.g_Na[None,...] * h * m**2.0 * (-self.E_Na[None,...] + V)))

        # V, h
        J[...,0::4,2::4] = torch.diag_embed(-1.0 / self.C[None,...] * (
                self.g_Na[None,...] * m**3.0 * (-self.E_Na[None,...] + V)))

        # V, n
        J[...,0::4,3::4] = torch.diag_embed(-1.0 / self.C[None,...] * (
                4.0 * self.g_K[None,...] * n**3.0 * (-self.E_K[None,...] + V)))

        # m, m
        J[...,1::4,1::4] = torch.diag(-1.0 / self.tau_m)

        # h, h 
        J[...,2::4,2::4] = torch.diag(-1.0 / self.tau_h)

        # n, n
        J[...,3::4,3::4] = torch.diag(-1.0 / self.tau_n)
    
        return J


class MassDamperSpring(torch.nn.Module):
    """
    Mass, spring, damper system of arbitrary size

    I use simple substitution to solve for the velocity and displacement
    in a first order system, so the size of the system is twice the size of the
    number of elements.

    Args:
        half_size (int): half size of problem

    Keyword args:
        K_range (tuple): (min stiffness, max stiffness)
        C_range (tuple): (min damping, max damping)
        M_range (tuple): (min mass, max mass)
        t_max (float): final time
        force_mag (float): magnitude of applied force
        force_period_range (tuple): range of periods for sin force

    """
    def __init__(self, half_size, K_range = (0.01,1.0), C_range = (1.0e-6, 1.0e-4),
            M_range = (1.0e-7, 1.0e-5), t_max = 1.0, force_mag = 1.0,
            force_period_range = (1.0e-2, 1.0)):
        super().__init__()

        self.half_size = half_size
        self.size = self.half_size * 2

        self.K = torch.nn.Parameter(torch.linspace(*K_range, self.half_size))
        self.C = torch.nn.Parameter(torch.linspace(*C_range, self.half_size))
        self.M = torch.nn.Parameter(torch.linspace(*M_range, self.half_size))
        
        self.t_max = t_max

        self.force_mag = force_mag
        self.force_period_range = force_period_range

    def setup(self, nbatch, ntime):
        """Setup and return time values and initial conditions for a given batch

        Args:
            nbatch (int): batch size
            ntime (int): number of time steps
        """
        time = torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        y0 = torch.zeros(nbatch, self.size)

        return time, y0

    def force(self, t):
        """Return the driving force

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
        """
        return self.force_mag * torch.sin(
                2.0 * torch.pi / torch.linspace(*self.force_period_range, t.shape[-1],
                    device = t.device) * t,) 

    def rate(self, t, y, force):
        """
        Return the state rate

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
        """
        ydot = torch.zeros_like(y)

        # Separate out for convenience
        u = y[...,:self.half_size]
        v = y[...,self.half_size:]
        
        # Differences
        du = torch.diff(u, dim = -1)
        dv = torch.diff(v, dim = -1)

        # Rate
        # Velocity
        ydot[...,:self.half_size] = v
        
        # Springs
        ydot[...,self.half_size:-1] += self.K[...,:-1] * du / self.M[...,:-1]
        ydot[...,self.half_size+1:] += -self.K[...,:-1] * du / self.M[...,1:]
        ydot[...,-1] += -self.K[...,-1] * u[...,-1] / self.M[...,-1]
        ydot[...,self.half_size] += force / self.M[...,0]

        # Dampers
        ydot[...,self.half_size:-1] += self.C[...,:-1] * dv / self.M[...,:-1]
        ydot[...,self.half_size+1:] += -self.C[...,:-1] * dv / self.M[...,1:]
        ydot[...,-1] += -self.C[...,-1] * v[...,-1] / self.M[...,-1]

        return ydot

    def jacobian(self, t, y, force):
        """
        Return the problem jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """
        J = torch.zeros(y.shape + (self.size,), device = t.device)

        # Derivative
        # Velocity 
        J[...,:self.half_size,self.half_size:] += torch.eye(self.half_size, device = t.device)

        # Springs
        J[...,self.half_size:,:self.half_size] += torch.diag_embed(-self.K / self.M)
        J[...,self.half_size+1:,1:self.half_size] += torch.diag_embed(-self.K[...,:-1] / self.M[...,1:])
        J[...,self.half_size:,:self.half_size] += torch.diag_embed(self.K[...,:-1] / self.M[...,:-1], offset = 1)
        J[...,self.half_size:,:self.half_size] += torch.diag_embed(self.K[...,:-1] / self.M[...,1:], offset = -1)

        # Dampers
        J[...,self.half_size:,self.half_size:] += torch.diag_embed(-self.C / self.M)
        J[...,self.half_size+1:,self.half_size+1:] += torch.diag_embed(-self.C[...,:-1] / self.M[...,1:])
        J[...,self.half_size:,self.half_size:] += torch.diag_embed(self.C[...,:-1] / self.M[...,:-1], offset = 1)
        J[...,self.half_size:,self.half_size:] += torch.diag_embed(self.C[...,:-1] / self.M[...,1:], offset = -1)

        return J

def mbmm(A1, A2):
    """
    Batched matrix-matrix multiplication with several batch dimensions
    """
    return torch.einsum("...ik,...kj->...ij", A1, A2)

def dtanh(x):
    """Derivative of ReLU with respect to input
    """
    return 1.0 - torch.tanh(x)**2.0

class LinearNetwork(torch.nn.Module):
    """Linear + RELU network that I have laboriously worked out the analytical Jacobian :)

    Args:
        n (int): size of system - 1
        nlayers (int): number of linear layers - 1, i.e. nlayers = 1 means 2 layers
        t_max (float): final time
        force_mag (float): magnitude of applied force
        force_period_range (tuple): range of periods for sin force
    """
    def __init__(self, n, nlayers = 3, t_max = 1.0, force_mag = 1.0,
            force_period_range = (1.0e-2, 1.0)):
        super().__init__()
        
        self.n = n
        self.size = self.n + 1

        self.nlayers = nlayers

        self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(self.size, self.size) for i in range(self.nlayers)]
                + [torch.nn.Linear(self.size, self.n)])
        self.activation = torch.nn.Tanh()
        self.dactivation = dtanh
        
        self.t_max = t_max
        self.force_mag = force_mag
        self.force_period_range = force_period_range

    def setup(self, nbatch, ntime):
        """Setup and return time values and initial conditions for a given batch

        Args:
            nbatch (int): batch size
            ntime (int): number of time steps
        """
        time = torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        y0 = torch.zeros(nbatch, self.n)

        return time, y0

    def force(self, t):
        """Return the driving force

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
        """
        return self.force_mag * torch.sin(
                2.0 * torch.pi / torch.linspace(*self.force_period_range, t.shape[-1],
                    device = t.device) * t,) 

    def rate(self, t, y, force):
        """
        Return the state rate

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
        """
        x = torch.cat([y, force.unsqueeze(-1)], dim = -1)

        for l in self.layers:
            x = self.activation(l(x))

        return x

    def jacobian(self, t, y, force):
        """
        Return the problem jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state
            force (torch.tensor): tensor of driving forces

        Returns:
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """
        # Hmm, we have to track the forward pass
        x = torch.cat([y, force.unsqueeze(-1)], dim = -1)
        
        # Jacobian shape
        full_shape = y.shape[:-1] + (self.size,)
        J = torch.diag_embed(torch.ones(full_shape, device = y.device), 
                dim1 = -1, dim2 = -2)

        for l in self.layers:
            x = mbmm(x.unsqueeze(-2), l.weight.transpose(-1,-2)).squeeze(-2) + l.bias
            J = mbmm(mbmm(J,l.weight.transpose(-1,-2)), torch.diag_embed(self.dactivation(x), 
                    dim1 = -1, dim2 = -2))
            x = self.activation(x)
        
        # I suppose I could just untranspose everything...
        return J[...,:-1,:].transpose(-1,-2)
