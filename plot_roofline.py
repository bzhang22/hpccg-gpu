import matplotlib.pyplot as plt
import numpy as np

# Machine Specs
b200_bw = 8000.0   # GB/s
b200_peak_fp64 = 45000.0 # GFLOPS
b200_peak_fp32 = 90000.0 # GFLOPS

l4_bw = 300.0      # GB/s
l4_peak_fp64 = 470.0    # GFLOPS
l4_peak_fp32 = 30000.0  # GFLOPS

# Kernel Specs
ai_fp64 = 0.157 # 54 FLOPs / 344 Bytes
ai_fp32 = 0.237 # 54 FLOPs / 228 Bytes

b200_fp64_perf = 647.3
b200_fp32_perf = 961.8

l4_fp64_perf = 22.6
l4_fp32_perf = 41.4

# Plot Data
intensities = np.logspace(-3, 3, 1000)
b200_roof_fp64 = np.minimum(b200_peak_fp64, intensities * b200_bw)
b200_roof_fp32 = np.minimum(b200_peak_fp32, intensities * b200_bw)
l4_roof_fp64 = np.minimum(l4_peak_fp64, intensities * l4_bw)
l4_roof_fp32 = np.minimum(l4_peak_fp32, intensities * l4_bw)

plt.figure(figsize=(12, 8))

# Plot B200 Roofline
plt.plot(intensities, b200_roof_fp64, label='B200 Roofline (8 TB/s) [FP64 Peak]', color='#76b900', linewidth=2, linestyle='solid')
plt.plot(intensities, b200_roof_fp32, label='B200 Roofline (8 TB/s) [FP32 Peak]', color='#76b900', linewidth=2, linestyle='dashed')

# Plot L4 Roofline
plt.plot(intensities, l4_roof_fp64, label='L4 Roofline (300 GB/s) [FP64 Peak]', color='#0071c5', linewidth=2, linestyle='solid')
plt.plot(intensities, l4_roof_fp32, label='L4 Roofline (300 GB/s) [FP32 Peak]', color='#0071c5', linewidth=2, linestyle='dashed')

# Plot SpMV Points
plt.scatter([ai_fp64], [b200_fp64_perf], color='darkred', s=150, zorder=5, edgecolor='black', marker='o', label='B200 SpMV (FP64: 647 GFLOPS)')
plt.scatter([ai_fp32], [b200_fp32_perf], color='red', s=150, zorder=5, edgecolor='black', marker='^', label='B200 SpMV (FP32: 962 GFLOPS)')

plt.scatter([ai_fp64], [l4_fp64_perf], color='darkblue', s=150, zorder=5, edgecolor='black', marker='o', label='L4 SpMV (FP64: 22.6 GFLOPS)')
plt.scatter([ai_fp32], [l4_fp32_perf], color='blue', s=150, zorder=5, edgecolor='black', marker='^', label='L4 SpMV (FP32: 41.4 GFLOPS)')

# Annotations
plt.axvline(x=ai_fp64, color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=ai_fp32, color='gray', linestyle=':', alpha=0.7)
plt.text(ai_fp64 * 0.8, 2, 'FP64 AI\n(0.157)', rotation=90, va='bottom', ha='right', fontsize=11, color='black', alpha=0.8)
plt.text(ai_fp32 * 1.1, 2, 'FP32 AI\n(0.237)', rotation=90, va='bottom', ha='left', fontsize=11, color='black', alpha=0.8)

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
plt.fill_between(intensities, 0, b200_roof_fp64, where=(intensities < (b200_peak_fp64/b200_bw)), color='#76b900', alpha=0.1)

plt.tight_layout()
plt.savefig('roofline_plot.png', dpi=300)
print("Saved roofline_plot.png")
