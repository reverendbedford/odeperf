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
       
        plt.figure()
        plt.plot(data.nchunk, data.isel(nsize=-1, nbatch=-1, jac_type=0, backward_type = 0).total_time)
        plt.legend(["Thomas", "PCR"], loc = 'best')
        plt.xticks([0,250,500,750,1000])
        plt.xlabel("$n_{chunk}$")
        plt.ylabel("Wall time (s)")
        plt.tight_layout()
        plt.savefig("%s_time_biggest.pdf" % model)

        print(np.max(data.isel(jac_type=0, backward_type = 0, nchunk = 0).total_time / data.isel(jac_type=0, backward_type = 0, nchunk = -1).total_time))

        plt.figure()
        plt.plot(data.nchunk, data.isel(nsize=-1, nbatch=-1, jac_type=0, backward_type = 0).memory_use/2**20)
        plt.legend(["Thomas", "PCR"], loc = 'best')
        plt.xticks([0,250,500,750,1000])
        plt.xlabel("$n_{chunk}$")
        plt.ylabel("Max memory (MB)")
        plt.tight_layout()
        plt.savefig("%s_memory_biggest.pdf" % model)

        plt.figure()
        plt.plot(data.nchunk, data.isel(nsize=0, nbatch=-1, jac_type=0, backward_type = 0).total_time)
        plt.legend(["Thomas", "PCR"], loc = 'best')
        plt.xticks([0,250,500,750,1000])
        plt.xlabel("$n_{chunk}$")
        plt.ylabel("Wall time (s)")
        plt.tight_layout()
        plt.savefig("%s_time_smallest.pdf" % model)

        plt.figure()
        plt.plot(data.nchunk, data.isel(nsize=0, nbatch=-1, jac_type=0, backward_type = 0).memory_use/2**20)
        plt.legend(["Thomas", "PCR"], loc = 'best')
        plt.xticks([0,250,500,750,1000])
        plt.xlabel("$n_{chunk}$")
        plt.ylabel("Max memory (MB)")
        plt.tight_layout()
        plt.savefig("%s_memory_smallest.pdf" % model)
