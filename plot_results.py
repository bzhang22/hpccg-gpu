import matplotlib.pyplot as plt
import numpy as np

# Data
grids = [64, 128, 192, 256]
gpu_times = [0.111, 0.862, 3.012, 7.201]

cpu_grids = [192, 256]
cpu_times = [158.2, 384.2]

# 1. Scaling Plot
plt.figure(figsize=(8, 6))
plt.plot(grids, gpu_times, marker='o', linewidth=2, color='green', label='CUDA GPU (NVIDIA L4)')
plt.plot(cpu_grids, cpu_times, marker='s', linewidth=2, color='red', linestyle='--', label='16-Thread CPU (OpenMP)')

plt.title('Performance Scaling: CPU vs GPU\n(Grid Size vs Total Execution Time)', fontsize=14, fontweight='bold')
plt.xlabel('Grid Dimension (N x N x N)', fontsize=12)
plt.ylabel('Execution Time (Seconds)', fontsize=12)
plt.xticks(grids)
plt.yscale('log') # Log scale is better to show the massive difference
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('scaling_plot.png', dpi=300)
print("Saved scaling_plot.png")

# 2. Kernel Time Pie Chart
# Using data from 256^3:
# Total: 7.20s, DDOT: 0.26s, WAXPBY: 0.71s, SPARSEMV: 5.96s
labels = ['Sparse MV (SPARSEMV)', 'Vector Updates (WAXPBY)', 'Dot Product (DDOT)']
times = [5.95684, 0.71448, 0.262824]
colors = ['#ff9999','#66b3ff','#99ff99']

plt.figure(figsize=(7, 7))
plt.pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
        explode=(0.05, 0, 0), shadow=True, textprops={'fontsize': 12})
plt.title('GPU Kernel Execution Time Breakdown\n(Grid: 256x256x256)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_pie.png', dpi=300)
print("Saved kernel_pie.png")
