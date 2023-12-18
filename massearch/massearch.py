import numpy as np
import pandas as pd
from time import time
from numba import cuda
import math
import random
import csv


def cpu_mass_search(N, H, R):
    for j in range(R.shape[1]):
        for i in range(R.shape[0]):
            n = N[i]
            for k in range(len(n)):
                 if n[k] == H[j] and j - k >= 0:
                     R[i, j - k] -= 1
    return R


@cuda.jit
def gpu_search_kernel(N, H, R):
    i, j = cuda.grid(2)
    if i < N.shape[0] and j < H.shape[0]:
        n = N[i]
        h = H[j]
        for k in range(n.size):
            if n[k] == h and j - k >= 0:
                cuda.atomic.sub(R, (i, j - k), 1)

def gpu_mass_search(N, H, R):
    threadsperblock = (16, 16)
    blockspergrid_x = (N.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (H.shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_N = cuda.to_device(N)
    d_H = cuda.to_device(H)
    d_R = cuda.to_device(R)
    startime = cuda.event()
    endp = cuda.event()
    startime.record()
    gpu_search_kernel[blockspergrid, threadsperblock](d_N, d_H, d_R)
    endp.record()
    endp.synchronize()
    elapsedp = cuda.event_elapsed_time(startime, endp)
    return d_R.copy_to_host(), elapsedp / 1000


def save_array_to_csv(my_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(my_array)

sizes = [[50, 25], [70, 35], [100, 50], [300,150], [600, 300]]

str_sz = 5

ABC = 256

df = pd.DataFrame({"CPU time": pd.Series(dtype='float'),
                   "GPU time": pd.Series(dtype='float'),
                   "ACC": pd.Series(dtype='float'),
                   "CPU_R == GPU_R": pd.Series(dtype='bool')})

for size in sizes:

  N = np.random.randint(ABC, size=(size[0], str_sz), dtype=np.int64)
  H = np.random.randint(ABC, size=size[1], dtype=np.int64)
  R = np.full((size[0], size[1]), fill_value=0)

  gpu_R, gpu_time = gpu_mass_search(N, H, R)

  start = time()
  cpu_R = cpu_mass_search(N, H, R)
  cpu_time = time() - start

  ind = f"N={size[0]}:H={size[1]}"

  df.loc[ind, "GPU time"] = gpu_time
  df.loc[ind, "CPU time"] = cpu_time
  df.loc[ind, "CPU_R == GPU_R"] = np.array_equal(gpu_R, cpu_R)

save_array_to_csv(gpu_R,"gpu_csv.csv")
df["ACC"] = df["CPU time"] / df["GPU time"]
print(df)