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
    nsize = [1,2,3,5,10,15,20,25,50]
    nbatch = [3,10,30,100]
    ntime = 300
    nchunk = [1,2,3,4,5,10,20,30,40,50,75,100]
    jac_type = ["analytic", "AD-backward", "AD-forward"]
    backward_type = ["adjoint", "AD"]
    integration_method = "backward-euler"

    res = test.run_grid(model, nsize, nbatch, ntime, nchunk, jac_type, 
            backward_type, integration_method, device, repeat = repeats)

    res.to_netcdf(name + "_" + dtype + "_" + devtype + ".nc")

def run_neuron():
    name = "neuron"
    repeats = 3

    model = samples.Neuron
    nsize = [1,2,3,4,5,6,7,8,9,10]
    nbatch = [3,10,30,100]
    ntime = 300
    nchunk = [1,2,3,4,5,10,20,30,40,50,75,100]
    jac_type = ["analytic", "AD-backward", "AD-forward"]
    backward_type = ["adjoint", "AD"]
    integration_method = "backward-euler"

    res = test.run_grid(model, nsize, nbatch, ntime, nchunk, jac_type, 
            backward_type, integration_method, device, repeat = repeats)

    res.to_netcdf(name + "_" + dtype + "_" + devtype + ".nc")

def run_neural_network():
    name = "nn"
    repeats = 3

    model = samples.LinearNetwork
    nsize = [1,5,10,15,20,25,50,100]
    nbatch = [3,10,30,100]
    ntime = 300
    nchunk = [1,2,3,4,5,10,20,30,40,50,75,100]
    jac_type = ["analytic", "AD-backward", "AD-forward"]
    backward_type = ["adjoint", "AD"]
    integration_method = "backward-euler"

    res = test.run_grid(model, nsize, nbatch, ntime, nchunk, jac_type, 
            backward_type, integration_method, device, repeat = repeats)

    res.to_netcdf(name + "_" + dtype + "_" + devtype + ".nc")


if __name__ == "__main__":
    #run_massdamperspring()
    #run_neuron()
    run_neural_network()



