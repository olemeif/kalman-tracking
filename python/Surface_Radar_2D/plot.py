import matplotlib.pyplot as plt
import numpy as np

def plot_simulation(t, s_true, v_true, a_true, s_measured, v_diff):
    """
    Plot the simulated ground truth and measured x-y-position data for the 2D surface radar scenario.
    Args:
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector (2D)
        v_true: np.ndarray, true velocity vector (2D)
        a_true: np.ndarray, true acceleration vector (2D)
        s_measured: np.ndarray, measured position vector (2D)
        v_diff: np.ndarray, estimated velocity vector using finite difference (2D)
    """

    plt.plot(s_true[:, 0], s_true[:, 1], label='True Position', color='blue')
    plt.scatter(s_measured[:, 0], s_measured[:, 1], label='Measured Position', color='red', s=10)
    plt.title('2D Surface Radar Simulation')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.legend()
    plt.axis('equal')

    plt.show()
