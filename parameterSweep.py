import numpy as np
import matplotlib.pyplot as plt
import time

# Define a basic 5-point stencil Laplacian function
def lapl(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 
        4 * grid
    )

def initiliase(len, option):
    if option == 0:
        gridu = np.random.rand(len, len)
        gridv = np.random.rand(len, len)
    elif option == 1:
        gridu = np.ones((len, len))
        gridv = np.zeros((len, len))

        # initial concentrations at center
        low = (len // 2) - 9
        high = (len // 2) + 10
        gridu[low:high, low:high] = 0.5 + np.random.uniform(0, 0.1, (19, 19))
        gridv[low:high, low:high] = 0.25 + np.random.uniform(0, 0.1, (19, 19))
    else:
        gridu = np.random.rand(len, len)
        gridv = np.random.rand(len, len)

    return gridu, gridv

def turingP(Du=0.16, Dv=0.08, f=0.035, k=0.064):
    L = 100         # Grid size
    t_steps = 10000   # Total simulation frames for the GIF
    #Du = 0.16 #Diffusion rate of activator u
    #Dv = 0.08 #Diffusion rate of inhibitor v
    #f, k = 0.035, 0.064
    dt = 1.0

    # choose options
    # 0 - random
    # 1 - blob in the middle
    initu, initv = initiliase(L, 1)

    ti = 0
    stepDiff = 1
    while ti < t_steps and stepDiff != 0:
        u, v = initu, initv
        lu, lv = lapl(u), lapl(v)

        initu = u + (Du * lu + f * (1.0 - u) - u * v * v) * dt
        initv = v + (Dv * lv + u * v * v - (f + k)* v) * dt

        stepDiff = np.sum(np.abs(initu - u))

        ti = ti + 1

    #print("Time steps:" + str(ti))
    return ti

f_vals = np.arange(0.01,0.06,0.001)
k_vals = np.arange(0.03,0.07,0.001)

t_outs = np.zeros((len(f_vals), len(k_vals)))

start_time = time.perf_counter()

# Sweep over parameter space
for i, f in enumerate(f_vals):
    for j, k in enumerate(k_vals):
        t_outs[i, j] = turingP(f=f, k=k)

end_time = time.perf_counter()
tot_time = end_time - start_time

print(f"Execution time: {tot_time:.6f} seconds")
np.save("turing_sweep_data.npy", t_outs)
np.save("f_vals.npy", f_vals)
np.save("k_vals.npy", k_vals)
print("Sweep complete and saved to disk.")


plot_bounds = [k_vals.min(), k_vals.max(), f_vals.min(), f_vals.max()]

plt.imshow(t_outs, extent=plot_bounds, origin='lower', aspect='auto', cmap='viridis')

plt.colorbar(label='Survival Time')
plt.xlabel('Kill Rate (k)')
plt.ylabel('Feed Rate (f)')

plt.show()