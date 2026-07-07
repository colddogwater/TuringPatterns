import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define Constants & Grid Dimensions ---
W, L = 100, 500 # Width and height
#L = 100         # Grid size
t_steps = 1200   # Total simulation frames for the GIF
Du = 0.16 #Diffusion rate of activator u
Dv = 0.08 #Diffusion rate of inhibitor v
f, k = 0.04, 0.063
dt = 1.0

# Define a basic 5-point stencil Laplacian function
def lapl(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 
        4 * grid
    )


def update(frame):
    global initu, initv
    global Du, Dv, f, k, dt

    for _ in range(10):
        u, v = initu, initv
        lu, lv = lapl(u), lapl(v)

        initu = u + (Du * lu + f * (1.0 - u) - u * v * v) * dt
        initv = v + (Dv * lv + u * v * v - (f + k)* v) * dt

    im_u.set_array(initu)

    return(im_u)

def initiliase(W, L, option):
    if option == 0:
        gridu = np.random.rand(L, W)
        gridv = np.random.rand(L, W)
    elif option == 1:
        gridu = np.ones((L, W))
        gridv = np.zeros((L, W))

        # initial concentrations at center
        lowL = (L // 2) - 9
        highL = (L // 2) + 10
        lowW = (W // 2) - 9
        highW = (W // 2) + 10
        gridu[lowL:highL, lowW:highW] = 0.5 + np.random.uniform(0, 0.1, (19, 19))
        gridv[lowL:highL, lowW:highW] = 0.25 + np.random.uniform(0, 0.1, (19, 19))
    else:
        gridu = np.random.rand(L, W)
        gridv = np.random.rand(L, W)

    return gridu, gridv

   
# choose options
# 0 - random
# 1 - blob in the middle
initu, initv = initiliase(L, W, 1)

im_u = plt.imshow(initu, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

fig = plt.gcf()
ani = FuncAnimation(fig, update, frames=t_steps, blit=True, interval=40)

ani.save('reaction_diffusion.gif', writer='pillow', fps=25)
plt.close()

print("Finished successfully")