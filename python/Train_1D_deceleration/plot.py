import matplotlib.pyplot as plt
import numpy as np

def plot_all(t, s_true, s_measured, v_true, v_diff, sigma_m):
    plt.figure(figsize=(12, 8))

    # Position
    plt.subplot(2, 2, 1)
    plt.plot(t, s_true, label="True Position", linewidth=2)
    plt.plot(t, s_measured, label="Measured Position", linestyle="--", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.title(f"Position with Noise (σ = {sigma_m:.2f} m)")

    # Velocity
    plt.subplot(2, 2, 2)
    plt.plot(t, v_true, label="True Velocity", linewidth=2)
    plt.plot(t, v_diff, label="Estimated Velocity (Finite Difference)", linestyle="--", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.title("Velocity Estimation")

    # Error
    plt.subplot(2, 2, 3)
    error = s_measured - s_true
    plt.plot(t, error)
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title("Measurement Error")

    # Error Statistics
    plt.subplot(2, 2, 4)
    plt.hist(error, bins=30, alpha=0.7)
    plt.xlabel("Error (m)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")

    plt.tight_layout()
    plt.show()