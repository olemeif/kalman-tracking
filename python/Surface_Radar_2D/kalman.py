import numpy as np
from scipy.linalg import expm

from kalman_common import correct, predict

def system_matrices(dt_s: float):
    """
    System matrices for a 2D constant velocity model with state x = [x, y, v_x, v_y].
    Each axis is an independent kinematic model 2nd order, [1, Section 12.1 / 12.2]:
        x_dot = A x + G z,   y = C x,   z = [a_x, a_y] unknown acceleration
    Speed and direction changes are not modelled, they enter through the process noise z.
    Args:
        dt_s: Time step in seconds
    Returns:
        A_d: Discrete system matrix
        G_d: Discrete process noise matrix (noise held constant over one step)
        C: Output matrix
    """
    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))
    A = np.block([[Z2, I2], [Z2, Z2]])  # [1, Eq. 12.2] per axis
    G = np.vstack([Z2, I2])             # [1, Eq. 12.2] per axis
    C = np.hstack([I2, Z2])             # [1, Eq. 12.3] per axis, radar measures x and y

    n, m = A.shape[0], G.shape[1]

    # expm([[A, G], [0, 0]] * dt) = [[Ad, Gd], [0, I]]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = G
    Md = expm(M * dt_s)

    A_d = Md[:n, :n]   # expm(A*dt_s), [1, Eq. 1.10 / 12.4]
    G_d = Md[:n, n:]   # integral from 0 to dt_s of expm(A*tau) @ G dtau, [1, Eq. 8.12 / 12.12]

    return A_d, G_d, C

def kalman(s_measured, dt_s, sigma_m, v_0, Q):
    """
    Kalman filter for 2D surface radar tracking of a single target with unknown velocity.
    Speed and direction may change; both are tracked by the velocity states.
    Args:
        s_measured: np.ndarray, measured position [x, y] (n_steps, 2)
        dt_s: float, time step in seconds
        sigma_m: float, measurement noise standard deviation per axis
        v_0: [v_x, v_y], initial velocity estimate (used for the initial prior)
        Q: float, process noise Var(z_v) per axis in (m/s)^2 per step
    Returns:
        x_kf: np.ndarray, corrected state estimates x_tilde(k) = [x, y, v_x, v_y] (n_steps, 4)
        P_kf: np.ndarray, corrected covariances P_tilde(k) (n_steps, 4, 4)
        K_kf: np.ndarray, Kalman gains (n_steps, 4, 2)
    """
    n_steps = s_measured.shape[0]
    A_d, G_d, C = system_matrices(dt_s)

    # Uncorrelated noise on both axes, [1, Eq. 1.20]
    R = sigma_m ** 2 * np.eye(2)

    # Partially constant noise, [1, Eq. 12.17]: Var(z_a) = Var(z_v) / dt^2, so that per axis
    # G_d Q G_d^T = [[dt^2/4, dt/2], [dt/2, 1]] * Var(z_v)
    Q_a = Q / dt_s ** 2 * np.eye(2)
    GQG = G_d @ Q_a @ G_d.T                     # [1, Eq. 5.2]

    # State vector: [x_position, y_position, x_velocity, y_velocity]
    x_kf = np.zeros((n_steps, 4))
    P_kf = np.zeros((n_steps, 4, 4))
    K_kf = np.zeros((n_steps, 4, 2))

    # Initial prior for time k = 0. The filter must not use the ground truth.
    # The velocity is unknown, so it gets a large variance, [1, Section 1.4].
    x_hat = np.array([[s_measured[0, 0]], [s_measured[0, 1]], [v_0[0]], [v_0[1]]])
    P_hat = np.diag([100.0 ** 2, 100.0 ** 2, 20.0 ** 2, 20.0 ** 2])

    for k in range(n_steps):
        # Correction, [1, Eq. 5.3 - 5.5]
        x_tilde, P_tilde, K = correct(x_hat, P_hat, C, R, s_measured[k][:, None])

        # Store the corrected estimate for time k
        x_kf[k] = x_tilde[:, 0]
        P_kf[k] = P_tilde
        K_kf[k] = K

        # Prediction for time k+1, [1, Eq. 5.1 - 5.2]
        x_hat, P_hat = predict(x_tilde, P_tilde, A_d, GQG)

    return x_kf, P_kf, K_kf
