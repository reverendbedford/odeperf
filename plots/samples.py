#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    names = ["massdamperspring", "neuron", "nn", "chaboche"]

    plt.style.use('single')
    for n in names:
        times = torch.load("../%s_times.pt" % n)
        trajs = torch.load("../%s_trajectory.pt" % n)
        
        plt.figure()
        plt.plot(times.cpu()[...,0], trajs.detach().cpu()[...,0,:])
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.tight_layout()
        plt.savefig("%s_sample.pdf" % n)
