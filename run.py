#!/usr/bin/env python

import torch

import test, samples

# Comment for single precision
torch.set_default_tensor_type(torch.DoubleTensor)

# Setup appropriately...
device = torch.device("cuda")

if __name__ == "__main__":
    time_f, time_b, check = test.run_test_case(samples.MassDamperSpring,
            100, 50, 100, 10, "analytic", "adjoint", "backward-euler", 
            device)

    print(time_f, time_b, check)


