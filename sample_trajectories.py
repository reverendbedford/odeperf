#!/usr/bin/env python

import torch

import test, samples

# Comment for single precision
torch.set_default_tensor_type(torch.DoubleTensor)
dtype = "double"

# Setup appropriately...
device = torch.device("cuda:0")
devtype = "cuda"


def run_massdamperspring():
    name = "massdamperspring"
    model = samples.MassDamperSpring
    times, traj = test.run_test_case(model, 2, 1, 2000, 100, "analytic", 
            "thomas", "adjoint", "backward-euler", device, 
            return_full = True)
    torch.save(times, "%s_times.pt" % name)
    torch.save(traj, "%s_trajectory.pt" % name)


def run_neuron():
    name = "neuron"
    model = samples.Neuron
    times, traj = test.run_test_case(model, 1, 1, 2000, 100, "analytic", 
            "thomas", "adjoint", "backward-euler", device, 
            return_full = True)
    torch.save(times, "%s_times.pt" % name)
    torch.save(traj, "%s_trajectory.pt" % name)

def run_neural_network():
    name = "nn"
    model = samples.LinearNetwork
    times, traj = test.run_test_case(model, 4, 1, 2000, 100, "analytic", 
            "thomas", "adjoint", "backward-euler", device, 
            return_full = True)
    torch.save(times, "%s_times.pt" % name)
    torch.save(traj, "%s_trajectory.pt" % name)

def run_chaboche():
    name = "chaboche"
    model = samples.Chaboche
    times, traj = test.run_test_case(model, 2, 1, 2000, 100, "analytic", 
            "thomas", "adjoint", "backward-euler", device, 
            return_full = True)
    torch.save(times, "%s_times.pt" % name)
    torch.save(traj, "%s_trajectory.pt" % name)

if __name__ == "__main__":
    run_massdamperspring()
    run_neuron()
    run_neural_network()
    run_chaboche()



