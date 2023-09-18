#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.style.use('single')

    speeds = []
    
    for model in ["massdamperspring", "neuron", "chaboche", "nn"]:
        print(model)
        data = xr.load_dataset("../%s_double_cuda.nc" % model)

        print(data)
       
        plt.figure()
        plt.plot(data.nchunk, data.isel(nsize=-1, nbatch=-1, solver_type = 0, backward_type = 0).total_time[:,1:] / data.isel(nsize=-1, nbatch=-1, solver_type = 0, backward_type = 0).total_time[:,0])
        plt.legend(["AD: backward", "AD: forward"], loc = 'best')
        plt.xticks([0,250,500,750,1000])
        plt.xlabel("$n_{chunk}$")
        plt.ylabel("Wall time ratio")
        plt.tight_layout()
        plt.savefig("%s_time_jacobian.pdf" % model)

        plt.figure()
        plt.plot(data.nchunk, data.isel(nsize=-1, nbatch=-1, solver_type = 0, backward_type = 0).memory_use[:,1:] / data.isel(nsize=-1, nbatch=-1, solver_type = 0, backward_type = 0).memory_use[:,0])
        plt.legend(["AD: backward", "AD: forward"], loc = 'best')
        plt.xticks([0,250,500,750,1000])
        plt.xlabel("$n_{chunk}$")
        plt.ylabel("Memory ratio")
        plt.tight_layout()
        plt.savefig("%s_memory_jacobian.pdf" % model)
