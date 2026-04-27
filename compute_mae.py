import sys
import numpy as np

def compute_errors(file1, file2):
    try:
        x_cpu = np.loadtxt(file1)
        x_gpu = np.loadtxt(file2)
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    if len(x_cpu) != len(x_gpu):
        print(f"Error: Length mismatch! CPU: {len(x_cpu)}, GPU: {len(x_gpu)}")
        return
        
    diff = np.abs(x_cpu - x_gpu)
    mae = np.mean(diff)
    max_ae = np.max(diff)
    
    print(f"--- Error Analysis ---")
    print(f"Vector Length: {len(x_cpu)}")
    print(f"Mean Absolute Error (MAE): {mae:e}")
    print(f"Max Absolute Error (MaxAE): {max_ae:e}")
    
    # Check if they are exactly identical
    if max_ae == 0.0:
        print("Status: PERFECT MATCH (Bit-for-bit identical)")
    elif max_ae < 1e-12:
        print("Status: EXCELLENT MATCH (Within normal floating-point roundoff)")
    else:
        print("Status: WARNING (Differences exceed normal roundoff)")

if __name__ == "__main__":
    compute_errors("baseline/x_cpu.txt", "gpu/x_gpu.txt")
