# Small Library for a diffusion algorithm for a 2D numpy array

# Diffusion based on Von Neumann Neighbourhood
# Only checks the 4 adjacent cells
# 
# Parameters:
#   array =  2D numpy array
#   diff_rate = Rate at which the material diffuses
# Assumes:
#   array elements are floats
#   array is square
def von_diffusion_2D(array, diff_rate):
    L = len(array[0])
    temp_grid = array.copy()
    for k in range(L):
        for j in range(L):
            curr_val = array[k][j]
            # For our current cell, look a t neighbours to see if shit is flowing into us
            # Only have to look to the right and down, but check if we are on boundaries
            if (j == L-1 and k == L-1): # we are in bottom corner
                continue
            elif (j == L-1): # Right edge
                bot_diff = array[k+1][j] - curr_val

                temp_grid[k][j] += diff_rate * bot_diff
                temp_grid[k+1][j] -= diff_rate * bot_diff
            elif (k == L-1): # Bottom row

                right_diff = array[k][j+1] - curr_val

                temp_grid[k][j] += diff_rate * right_diff
                temp_grid[k][j+1] -= diff_rate * right_diff
            else: # Good section
                #check neighbours
                #if neighbour is larger then it will flow into current cell
                right_diff = array[k][j+1] - curr_val
                bot_diff = array[k+1][j] - curr_val

                temp_grid[k][j] += diff_rate * right_diff
                temp_grid[k][j+1] -= diff_rate * right_diff

                temp_grid[k][j] += diff_rate * bot_diff
                temp_grid[k+1][j] -= diff_rate * bot_diff

    grid = temp_grid.copy()

    return grid