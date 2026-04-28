import matplotlib.pyplot as plt
import numpy as np

# Machine Specs
b200_bw = 8000.0   # GB/s
b200_peak = 45000.0 # GFLOPS (Dense FP64 ceiling)

l4_bw = 300.0      # GB/s
l4_peak = 470.0    # GFLOPS (estimated FP64 ceiling)

# Kernel Specs
ai = 0.157 # Arithmetic Intensity (FLOPs/Byte)
b200_spmv_perf = 647.3 # GFLOPS
l4_spmv_perf = 22.6    # GFLOPS

# Plot Data
intensities = np.logspace(-3, 3, 1000)
b200_roof = np.minimum(b200_peak, intensities * b200_bw)
l4_roof = np.minimum(l4_peak, intensities * l4_bw)

plt.figure(figsize=(10, 7))

# Plot B200 Roofline
plt.plot(intensities, b200_roof, label='NVIDIA B200 Roofline (8 TB/s)', color='#76b900', linewidth=3)
plt.plot(intensities, l4_roof, label='NVIDIA L4 Roofline (300 GB/s)', color='#0071c5', linewidth=3, linestyle='--')

# Plot SpMV Points
plt.scatter([ai], [b200_spmv_perf], color='red', s=150, zorder=5, edgecolor='black', label='SpMV on B200 (647 GFLOPS)')
plt.scatter([ai], [l4_spmv_perf], color='orange', s=150, zorder=5, edgecolor='black', label='SpMV on L4 (22.6 GFLOPS)')

# Annotations
plt.axvline(x=ai, color='gray', linestyle=':', alpha=0.7)
plt.text(ai * 1.1, 1, 'Arithmetic Intensity = 0.157 FLOPs/Byte\n(Pure Memory-Bound)', rotation=90, va='bottom', ha='left', fontsize=11, color='black', alpha=0.8)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=14, fontweight='bold')
plt.ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
plt.title('Empirical Roofline Model: HPCCG SpMV Kernel', fontsize=16, fontweight='bold')

plt.xlim(1e-2, 1e2)
plt.ylim(1, 100000)

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=12, loc='upper left')

# Fill memory bound region
plt.fill_between(intensities, 0, b200_roof, where=(intensities < (b200_peak/b200_bw)), color='#76b900', alpha=0.1)

plt.tight_layout()
plt.savefig('roofline_plot.png', dpi=300)
print("Saved roofline_plot.png")
