import numpy as np
from PIL import Image


def diffusionStep(grid, temp_grid, curr_val, k, j, D):
    L = len(grid[0])
    # ========= DIFFUSION STEP
    # For our current cell, look a t neighbours to see if shit is flowing into us
    # Only have to look to the right and down, but check if we are on boundaries
    if (j == L-1 and k == L-1): # we are in bottom corner
        xd = 1
    elif (j == L-1): # Right edge
        bot_diff = grid[k+1][j] - curr_val

        temp_grid[k][j] +=  D * bot_diff
        temp_grid[k+1][j] -=  D * bot_diff 
    elif (k == L-1): # Bottom row

        right_diff = grid[k][j+1] - curr_val

        temp_grid[k][j] +=  D * right_diff 
        temp_grid[k][j+1] -=  D *right_diff 
    else: # Good section
        #check neighbours
        #if neighbour is larger then it will flow into current cell
        right_diff = grid[k][j+1] - curr_val
        bot_diff = grid[k+1][j] - curr_val

        temp_grid[k][j] +=  D * (right_diff + bot_diff)
        temp_grid[k][j+1] -=  D * right_diff 

        temp_grid[k+1][j] -=  D * bot_diff 

def lap5(f, h2):
    """
    Use a five-point stencil with periodic boundary conditions to approximate
    the Laplacian. The corresponding array slices for each component of the
    stencil are noted.
    """
    f = np.pad(f, 1, mode='wrap')

    left = f[1:-1, :-2]     # shift left for f(x - h, y)
    right = f[1:-1, 2:]     # shift right for f(x + h, y)
    down = f[2:, 1:-1]      # shift down for f(x, y - h)
    up = f[:-2, 1:-1]       # shift up for f(x, y + h)
    center = f[1:-1, 1:-1]  # center for f(x, y)

    fxy = (left + right + down + up - 4 * center) / h2

    return fxy

# ----- PARAMETERS
t = 60
L = 100
Du = 0.2
Dv = 0.1
dt = 1
f = 0.025
k = 0.056
# -----

# Create a sample 2D array (e.g., 100x100 grid of random pixel values)
#gridv = np.random.uniform(0, 1, size=(L, L))
#gridu = np.random.uniform(0, 1, size=(L, L))

# initialize arrays
gridu = np.ones((L, L))
gridv = np.zeros((L, L))

# initial concentrations at center
low = (L // 2) - 9
high = (L // 2) + 10
gridu[low:high, low:high] = 0.5 + np.random.uniform(0, 0.1, (19, 19))
gridv[low:high, low:high] = 0.25 + np.random.uniform(0, 0.1, (19, 19))

# for the gif
frames = []

# Create an empty RGB array of zeros
rgb_visual = np.zeros((L, L, 3), dtype=np.uint8)

# Map Species A to the Red channel (Index 0)

rgb_visual[:, :, 0] = (gridv * 256).astype(np.uint8)

# Map Species B to the Blue channel (Index 2)
rgb_visual[:, :, 2] = (gridu * 256).astype(np.uint8)

# Now pass this directly to Pillow!
img = Image.fromarray(rgb_visual, 'RGB')

img = img.resize((300, 300), Image.Resampling.NEAREST)
img.save("imgs/original.png")
frames.append(img)

for i in range(t):
    tempu = gridu.copy()
    tempv = gridv.copy()
    for k in range(L):
        for j in range(L):
            curr_u = gridu[k][j]
            curr_v = gridv[k][j]

            # ----- Diffusion Step
            #diffusionStep(gridu, tempu, curr_u, k ,j, Du)
            #diffusionStep(gridv, tempv, curr_u, k ,j, Dv)

            tempu[k][j] += (f * (1 - curr_u) - curr_u * curr_v**2) * dt
            tempv[k][j] += (curr_u * curr_v**2 - (f + k) * curr_v) * dt

    tempu += Du * lap5(gridu, 1)
    tempv += Dv * lap5(gridv, 1)

    gridu = tempu.copy()
    gridv = tempv.copy()

    print("Time step:" + str(i) + " the v sum is " + str(gridv.sum()))
    print("Time step:" + str(i) + " the u sum is " + str(gridu.sum()))

    rgb_visual = np.zeros((L, L, 3), dtype=np.uint8)
    # Map Species A to the Red channel (Index 0)
    rgb_visual[:, :, 0] = (gridv * 256).astype(np.uint8)

    # Map Species B to the Blue channel (Index 2)
    rgb_visual[:, :, 2] = (gridu * 256).astype(np.uint8)

    # Now pass this directly to Pillow!
    img = Image.fromarray(rgb_visual, 'RGB')

    img = img.resize((300, 300), Image.Resampling.NEAREST)
    img.save("imgs/"+str(i)+"t.png")
    frames.append(img)
            

frames[0].save(
    "diffusion.gif",
    save_all=True,
    append_images=frames[1:], # Append the rest of the list
    optimize=False,
    duration=200, # Milliseconds per frame (200ms = 5 frames per second)
    loop=0 # 0 means loop infinitely
)
print("GIF saved successfully!")
            

