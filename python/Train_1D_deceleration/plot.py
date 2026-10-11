import matplotlib.pyplot as plt
import numpy as np

def plot_all(t, s_true, s_measured, s_est, P_s, v_true, v_diff, v_est, P_v, a_true, a_est, P_a, sigma_m):
    """
    Plot the results of the 1D train scenario.
    Args:
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector
        s_measured: np.ndarray, measured position vector with noise
        s_est: np.ndarray, estimated position vector from Kalman filter
        P_s: np.ndarray, estimated position variance from Kalman filter
        v_true: np.ndarray, true velocity vector
        v_diff: np.ndarray, velocity vector from finite difference of measured position
        v_est: np.ndarray, estimated velocity vector from Kalman filter
        P_v: np.ndarray, estimated velocity variance from Kalman filter
        a_true: np.ndarray, true acceleration vector
        a_est: np.ndarray, estimated acceleration vector from Kalman filter
        P_a: np.ndarray, estimated acceleration variance from Kalman filter
        sigma_m: float, standard deviation of the measurement noise
    """
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(t, s_true, label='True Position', color='black')
    plt.scatter(t, s_measured, label='Measured Position', color='red', s=10)
    plt.plot(t, s_est, label='Estimated Position', color='blue')
    plt.fill_between(t, s_est - 2 * np.sqrt(P_s), s_est + 2 * np.sqrt(P_s), color='blue', alpha=0.2)
    plt.title('Position')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid()

    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(t, v_true, label='True Velocity', color='black')
    plt.plot(t[1:], v_diff[1:], label='Finite Difference Velocity', color='orange')
    plt.plot(t, v_est, label='Estimated Velocity', color='blue')
    plt.fill_between(t, v_est - 2 * np.sqrt(P_v), v_est + 2 * np.sqrt(P_v), color='blue', alpha=0.2)
    plt.title('Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid()

    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(t, a_true, label='True Acceleration', color='black')
    plt.plot(t, a_est, label='Estimated Acceleration', color='blue')
    plt.fill_between(t, a_est - 2 * np.sqrt(P_a), a_est + 2 * np.sqrt(P_a), color='blue', alpha=0.2)
    plt.title('Acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()