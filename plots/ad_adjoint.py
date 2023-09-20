#!/usr/bin/env python3

import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = xr.load_dataset("../nn_adjoint_comp_double_cuda.nc")
    
    plt.style.use('single')
    plt.figure()
    plt.plot(data.ntime, data.isel(nsize=0, nbatch=0, nchunk=0, jac_type=0, solver_type = 0).total_time)
    plt.legend(["AD", "Adjoint"], loc = 'best')
    plt.xticks([0,500,1000,1500,2000])
    plt.xlabel("$n_{time}$")
    plt.ylabel("Wall time (s)")
    plt.tight_layout()
    plt.savefig("ad_adjoint_time.pdf")

    plt.figure()
    plt.plot(data.ntime, data.isel(nsize=0, nbatch=0, nchunk=0, jac_type=0, solver_type = 0).memory_use/2**20)
    plt.legend(["AD", "Adjoint"], loc = 'best')
    plt.xticks([0,500,1000,1500,2000])
    plt.xlabel("$n_{time}$")
    plt.ylabel("Max memory (MB)")
    plt.tight_layout()
    plt.savefig("ad_adjoint_memory.pdf")
