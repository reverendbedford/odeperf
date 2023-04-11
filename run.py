#!/usr/bin/env python

import torch

import test, samples

# Comment for single precision
torch.set_default_tensor_type(torch.DoubleTensor)
dtype = "double"

# Setup appropriately...
device = torch.device("cuda")
devtype = "cuda"

def run_massdamperspring():
    name = "massdamperspring"
    repeats = 3

    model = samples.MassDamperSpring
    nsize = [2,4,6,10,20,30,40,50]
    nbatch = [3,10,30,100,300]
    ntime = 300
    nchunk = [1,2,3,4,5,10,20,30,40,50,75,100]
    jac_type = ["analytic", "AD"]
    backward_type = ["adjoint", "AD"]
    integration_method = "backward-euler"

    res = test.run_grid(model, nsize, nbatch, ntime, nchunk, jac_type, 
            backward_type, integration_method, device, repeat = repeats)

    res.to_netcdf(name + "_" + dtype + "_" + devtype + ".nc")

if __name__ == "__main__":
    run_massdamperspring()


