"""Sample problems to evaluate the performance of the ode integration and 
adjoint backward pass routines.
"""

import torch

class MassDamperSpring(torch.nn.Module):
    """
    Mass, spring, damper system of arbitrary size

    I use simple substitution to solve for the velocity and displacement
    in a first order system, so the size of the system is twice the size of the
    number of elements.

    Args:
        size (int): size of problem

    Keyword args:
        K_range (tuple): (min stiffness, max stiffness)
        C_range (tuple): (min damping, max damping)
        M_range (tuple): (min mass, max mass)
        t_max (float): final time
        force_mag (float): magnitude of applied force
        force_period_range (tuple): range of periods for sin force

    """
    def __init__(self, size, K_range = (0.01,1.0), C_range = (1.0e-6, 1.0e-4),
            M_range = (1.0e-7, 1.0e-5), t_max = 1.0, force_mag = 1.0,
            force_period_range = (1.0e-2, 1.0)):
        super().__init__()

        self.size = size
        self.half_size = self.size // 2

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

    def jacobian(self, t, y):
        """
        Return the problem jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state

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
