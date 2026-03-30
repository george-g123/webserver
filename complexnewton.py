import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numba import cuda
import math
import time
import cmath

@cuda.jit
def kernel(result_real, result_imag, brightness_out, x_min, x_max, y_min, y_max, max_iter, tol, alpha, beta):
    x_idx, y_idx = cuda.grid(2)
    
    height, width = result_real.shape
    if x_idx >= width or y_idx >= height:
        return
    
    real_part = x_min + (x_idx / width) * (x_max - x_min)
    imag_part = y_min + (y_idx / height) * (y_max - y_min)
    z = complex(real_part, imag_part)

    for i in range(max_iter):
        f_val = z**4 + z**2 + 1
        f_prime = 4*z**3 + 2*z

        if abs(f_prime) == 0:
            break

        delta = f_val / f_prime
        z = z - delta
        
        abs_diff = abs(delta)
        if abs_diff < tol:
            result_real[y_idx, x_idx] = z.real
            result_imag[y_idx, x_idx] = z.imag
            
            if abs_diff < 1e-15:
                abs_diff = 1e-15
            
            log_term = math.log(abs_diff)
            smooth_val = i - math.log2(-log_term) + alpha / 2.0
            
            b_val = smooth_val * beta
            if b_val > 1.0: b_val = 1.0
            if b_val < 0.0: b_val = 0.0
            
            brightness_out[y_idx, x_idx] = b_val
            return

    result_real[y_idx, x_idx] = 0.0
    result_imag[y_idx, x_idx] = 0.0
    brightness_out[y_idx, x_idx] = 0.0

x_min, x_max = -8, 8
y_min, y_max = -8, 8
N = 3000
convergence_delta = 1e-6
max_iterations = 100
alpha = 3
beta = 0.05
max_color_options = 20 

start_milli = int(time.time() * 1000)

roots_real = np.zeros((N, N), dtype=np.float64)
roots_imag = np.zeros((N, N), dtype=np.float64)
brightness_map = np.zeros((N, N), dtype=np.float64)

d_roots_real = cuda.to_device(roots_real)
d_roots_imag = cuda.to_device(roots_imag)
d_brightness = cuda.to_device(brightness_map)

threads_per_block = (16, 16)
blocks_per_grid_x = int(np.ceil(N / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(N / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

print(f"Launching GPU Kernel with Grid: {blocks_per_grid}")

newton_kernel[blocks_per_grid, threads_per_block](
    d_roots_real, 
    d_roots_imag, 
    d_brightness, 
    x_min, x_max, y_min, y_max, 
    max_iterations, 
    convergence_delta, 
    alpha, 
    beta
)
cuda.synchronize() # Wait for GPU to finish

roots_real = d_roots_real.copy_to_host()
roots_imag = d_roots_imag.copy_to_host()
brightness_map = d_brightness.copy_to_host()

Z_final = roots_real + 1j * roots_imag

converged_mask = (brightness_map > 0)

hsv_data = np.zeros((N, N, 3))

root_angles = np.angle(Z_final)

hsv_data[converged_mask, 0] = (root_angles[converged_mask] + np.pi) / (2 * np.pi)
# hsv_data[converged_mask, 0] = (Z_final[converged_mask].imag * 0.1) % 1.0

hsv_data[converged_mask, 1] = 1.0
hsv_data[converged_mask, 2] = brightness_map[converged_mask]

rgb_data = mcolors.hsv_to_rgb(hsv_data)

fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
ax.imshow(rgb_data, extent=[x_min, x_max, y_min, y_max], origin='lower', interpolation='antialiased')

ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title(f'Newton-Raphson convergence in the complex plane based on $f(z)=z^4+z^2+1$ with $\\alpha={alpha}$,\n$\\beta={beta}$, $N={N}$.')
ax.set_aspect('equal', adjustable='box')

max_legend_entries = 9
legend_elements = []

# Loop through the sorted roots. i_old is the unsorted index of the root and i_new is the new index after sorting.
# for i_new, i_old in enumerate(sorted_order[:max_legend_entries]):
#     root = unique_roots[i_old]
#     hue = ((mapping_table[i_old] % color_options) / color_options) % 1.0
#     color = mcolors.hsv_to_rgb([hue, 1.0, 1.0])

#     if abs(root.imag) < 1e-10: label = f'$z_{{{i_new+1}}} = {root.real:.3f}$'
#     elif abs(root.real) < 1e-10: label = f'$z_{{{i_new+1}}} = {root.imag:.3f}i$'
#     else:
#         sign = '+' if root.imag >= 0 else '-'
#         label = f'$z_{{{i_new+1}}} = {root.real:.3f} {sign} {abs(root.imag):.3f}i$'
    
#     legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='white', label=label))

#     # We only want to add the dot if the root is visible to us.
#     if (abs(root.real)<=x_max) and (abs(root.imag)<=y_max):
#         ax.plot(root.real, root.imag, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1, zorder=10)

# # Limit the number of displayed roots and display "..." if there are more.
# if len(unique_roots) > max_legend_entries: legend_elements.append(mpatches.Patch(color='gray', label=f"... (+{len(unique_roots)-max_legend_entries} more)"))

# ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, title='Roots')

end_milli = int(time.time() * 1000)
print("Total Runtime: "+str(int(end_milli - start_milli)/1000)+"s")

plt.savefig("GPU_Figure_5.png")