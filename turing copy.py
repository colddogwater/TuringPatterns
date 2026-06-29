import numpy as np
from PIL import Image

# ----- PARAMETERS
t = 50
L = 100
alpha = 0.2
dt = 1
# -----

# Create a sample 2D array (e.g., 100x100 grid of random pixel values)
grid = np.random.uniform(0, 256, size=(L, L))
# 1. Create the array with values spread across the 0-255 spectrum
#base_grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
#grid = base_grid * 25.0

# for the gif
frames = []

visual_grid = grid.astype(np.uint8)
img = Image.fromarray(visual_grid)
img = img.resize((300, 300), Image.Resampling.NEAREST)
img.save("imgs/original.png")
frames.append(img)

for i in range(t):
    temp_grid = grid.copy()
    for k in range(L):
        for j in range(L):
            curr_val = grid[k][j]

            # ========= DIFFUSION STEP
            # For our current cell, look a t neighbours to see if shit is flowing into us
            # Only have to look to the right and down, but check if we are on boundaries
            if (j == L-1 and k == L-1): # we are in bottom corner
                continue
            elif (j == L-1): # Right edge
                bot_diff = grid[k+1][j] - curr_val

                temp_grid[k][j] += alpha * bot_diff * dt
                temp_grid[k+1][j] -= alpha * bot_diff * dt
            elif (k == L-1): # Bottom row

                right_diff = grid[k][j+1] - curr_val

                temp_grid[k][j] += alpha * right_diff * dt
                temp_grid[k][j+1] -= alpha * right_diff * dt
            else: # Good section
                #check neighbours
                #if neighbour is larger then it will flow into current cell
                right_diff = grid[k][j+1] - curr_val
                bot_diff = grid[k+1][j] - curr_val

                temp_grid[k][j] += alpha * right_diff * dt
                temp_grid[k][j+1] -= alpha * right_diff * dt

                temp_grid[k][j] += alpha * bot_diff * dt
                temp_grid[k+1][j] -= alpha * bot_diff * dt


            # ========= EQUATION STEP
            temp_grid += (curr_val /256) * (1 -curr_val /256)**2 * 0.05


    grid = temp_grid.copy()

    print("Time step:" + str(t) + " the sum is " + str(temp_grid.sum()))
    visual_grid = grid.astype(np.uint8)
    img = Image.fromarray(visual_grid)
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
            



    




