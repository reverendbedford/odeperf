"""
Objects for running tests over sample ODE systems
"""

import time
import itertools

import numpy as np
import xarray as xr
import tqdm
import torch
from functorch import jacrev, vmap

from pyoptmat import ode

class AnalyticJacobian(torch.nn.Module):
    """Wrapper providing the Jacobian using the analytic function
    
    Args:
        base (torch.nn.Module): base module
    """
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, t, y):
        """
        Return the state rate and Jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """
        return self.base.rate(t, y, self.base.force(t)), self.base.jacobian(t, y)

class ADJacobian(torch.nn.Module):
    """Wrapper providing the Jacobian using AD
    
    Args:
        base (torch.nn.Module): base module
    """
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, t, y):
        """
        Return the state rate and Jacobian

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """ 
        force = self.base.force(t)
        rate = self.base.rate(t, y, force)
        
        return rate, vmap(vmap(jacrev(self.base.rate, argnums = 1)))(t, y, force)

def run_test_case(model, nsize, nbatch, ntime, nchunk, jac_type,
        backward_type, integration_method, device):
    """Run a single test case

    Args:
        model (returns torch.nn.Module): ODE to use
        nsize (int): size ("breadth") for the problem
        nbatch (int): number of batches
        ntime (int): number of time steps ("depth") for the problem
        nchunk (int): number of vectorized time steps for integration
        jac_type (string): jacobian choice: "analytic" or "AD"
        backward_type (string): backward pass type: "adjoint" or "AD"
        integration_method (string): integration method for pyoptmat.ode.odeint
    """
    if jac_type == "analytic":
        model = AnalyticJacobian(model(nsize)).to(device)
    elif jac_type == "AD":
        model = ADJacobian(model(nsize)).to(device)
    else:
        raise ValueError("Unknown Jacobian type %s" % jac_type)
    
    times, y0 = model.base.setup(nbatch, ntime)
    times = times.to(device)
    y0 = y0.to(device)
    
    torch.cuda.reset_max_memory_allocated()
    t1 = time.time()
    if backward_type == "adjoint":
        res = ode.odeint_adjoint(model, y0, times, method = integration_method,
                block_size = nchunk)
    elif backward_type == "AD":
        res = ode.odeint(model, y0, times, method = integration_method,
                block_size = nchunk)
    else:
        raise ValueError("Unknown backward pass type %s" % backward_type)

    t2 = time.time()

    R = torch.norm(res)
    R.backward()

    t3 = time.time()

    mem = torch.cuda.max_memory_allocated()

    return t2 - t1, t3 - t2, R.detach().cpu().numpy(), mem

def merge_in(tf, a, b):
    ia = iter(a)
    ib = iter(b)

    res = []
    for w in tf:
        if w:
            res.append(next(ia))
        else:
            res.append(next(ib))

    return res

def run_grid(model, nsize, nbatch, ntime, nchunk, jac_type,
        backward_type, integration_method, device, repeat = 1):
    """Run a big grid of simulations and save results in a data frame
    """
    meta_names = ["nsize", "nbatch", "ntime", "nchunk", "jac_type",
            "backward_type", "integration_method"]

    params = [nsize, nbatch, ntime, nchunk, jac_type, backward_type,
            integration_method]
    grid = [not np.isscalar(p) for p in params]
    sizes = [len(p) for g,p in zip(grid,params) if g]
    fixed_values = [p for g,p in zip(grid,params) if not g] 
    
    iterator = itertools.product(*tuple(p for g,p in zip(grid, params) if g))

    tf_res = np.empty(sizes)
    tb_res = np.empty(sizes)
    mem_use = np.empty(sizes)
    check_res = np.empty(sizes)
    
    total_size = np.prod(sizes)

    for i,x in tqdm.tqdm(enumerate(iterator), total = total_size):
        full = merge_in(grid, x, fixed_values)
        
        tfs = []
        tbs = []
        mems = []
        for r in range(repeat):
            tf, tb, check, mem = run_test_case(model, *full, device)
            tfs.append(tf)
            tbs.append(tb)
            mems.append(mem)


        ind = np.unravel_index(i, sizes)

        tf_res[ind] = np.mean(tf)
        tb_res[ind] = np.mean(tb)
        mem_use[ind] = np.mean(mems)
        check_res[ind] = check

    # Setup the xarray frame...
    dims = {m : p for g,m,p in zip(grid, meta_names,params) if g}
    attrs = {m : p for g,m,p in zip(grid, meta_names, params) if not g}
    attrs["repeats" ] = repeat

    ds = xr.Dataset(
            data_vars = {
                "forward_pass": (dims, tf_res),
                "backward_pass": (dims, tb_res),
                "check_sum": (dims, check_res),
                "memory_use": (dims, mem_use)
                },
            coords = dims,
            attrs = attrs)
    
    return ds
