import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define Constants & Grid Dimensions ---
L = 100         # Grid size
t_steps = 600   # Total simulation frames for the GIF
Du = 0.16 #Diffusion rate of activator u
Dv = 0.08 #Diffusion rate of inhibitor v
f, k = 0.035, 0.060
dt = 1.0

# Define a basic 5-point stencil Laplacian function
def lapl(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 
        4 * grid
    )

def initiliase(len, option):
    match option:
        case 0: # Random
            gridu = np.random.rand(len, len)
            gridv = np.random.rand(len, len)
        case 1: # Center dot
            gridu = np.ones((L, L))
            gridv = np.zeros((L, L))

            # initial concentrations at center
            low = (len // 2) - 9
            high = (len // 2) + 10
            gridu[low:high, low:high] = 0.5 + np.random.uniform(0, 0.1, (19, 19))
            gridv[low:high, low:high] = 0.25 + np.random.uniform(0, 0.1, (19, 19))
    return gridu, gridv

# choose options
# 0 - random
# 1 - blob in the middle
initu, initv = initiliase(L, 1)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

im_u = axs[0].imshow(initu, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

im_v = axs[1].imshow(initv, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

def update(frame):
    global initu, initv

    for _ in range(10):
        u, v = initu, initv
        lu, lv = lapl(u), lapl(v)

        initu = u + (Du * lu + f * (1.0 - u) - u * v * v) * dt
        initv = v + (Dv * lv + u * v * v - (f + k)* v) * dt

    im_u.set_array(initu)
    im_v.set_array(initv)

    return(im_u, im_v)

ani = FuncAnimation(fig, update, frames=t_steps, blit=True, interval=40)

ani.save('reaction_diffusion.gif', writer='pillow', fps=25)
plt.close()

print("Finished successfully")