import numpy as np

def kalman_3rd_degree(
    s_measured: np.ndarray,
    dt_s: float,
    sigma_m: float,
    v_0: float,
    Q: float
):
    """
    Kalman filter for a 3rd degree kinematic model (position, speed, acceleration) in 1D.
    As described in [1, Section 12.3].
    Args:
        s_measured: Measured position (1D array of length n_steps)
        dt_s: Time step in seconds
        sigma_m: Standard deviation of the position measurement noise
        v_0: Initial velocity estimate (used for the initial prior)
        Q: Process noise Var(z_a), the change in acceleration per step, in (m/s^2)^2
    Returns:
        x_est: Corrected state estimates x_tilde(k) = [s, v, a] (array of shape (n_steps, 3))
        P_est: Corrected covariances P_tilde(k), full 3x3 matrix (array of shape (n_steps, 3, 3))
        K_est: Kalman gains [K_s, K_v, K_a] (array of shape (n_steps, 3))
    """
    n_steps = len(s_measured)
    R = sigma_m ** 2  # [1, Eq. 1.18]

    A_d = np.array([[1, dt_s, 0.5*dt_s**2], [0, 1, dt_s], [0, 0, 1]])     # [1, Eq. 12.29]
    C = np.array([[1, 0, 0]])

    GQG = np.array([[1/4*dt_s**4, 1/2*dt_s**3, 1/2*dt_s**2],
                    [1/2*dt_s**3, dt_s**2,     dt_s],
                    [1/2*dt_s**2, dt_s,        1]]) * Q  # method 1: direct discretization, [1, p. 138]
    # Method 2, [1, Section 12.3] (swap in for the active GQG):
    # GQG = np.array([[1/36*dt_s**4, 1/12*dt_s**3, 1/6*dt_s**2], [1/12*dt_s**3, 1/4*dt_s**2, 1/2*dt_s], [1/6*dt_s**2, 1/2*dt_s, 1]])
    # Method 3, [1, Section 12.3]:
    # GQG = np.array([[1/20*dt_s**4, 1/8*dt_s**3, 1/6*dt_s**2], [1/8*dt_s**3, 1/3*dt_s**2, 1/2*dt_s], [1/6*dt_s**2, 1/2*dt_s, 1]])

    # Initial prior for time k = 0. The filter must not use the ground truth.
    # Position: rough guess with a large variance, so the first measurement dominates.
    x_hat = np.array([[s_measured[0]], [v_0], [0.0]])
    P_hat = np.array([[100.0 ** 2, 0, 0], [0, 20.0 ** 2, 0], [0, 0, 10.0 ** 2]])

    # Storage for the corrected estimates x_tilde(k), their covariances and the gains.
    # The full covariance is kept, because NEES needs the off-diagonal elements.
    x_est = np.zeros((n_steps, 3))
    P_est = np.zeros((n_steps, 3, 3))
    K_est = np.zeros((n_steps, 3))

    for k in range(n_steps):
        # Correction
        K = P_hat @ C.T @ np.linalg.inv(C @ P_hat @ C.T + R)    # [1, Eq. 12.38]
        x_tilde = x_hat + K @ (s_measured[k] - C @ x_hat)       # [1, Eq. 12.39]
        P_tilde = (np.eye(3) - K @ C) @ P_hat                   # [1, Eq. 12.40]

        # Store the corrected estimate for time k
        x_est[k] = x_tilde[:, 0]
        P_est[k] = P_tilde
        K_est[k] = K[:, 0]

        # Prediction for time k+1
        x_hat = A_d @ x_tilde                   # [1, Eq. 12.41]
        P_hat = A_d @ P_tilde @ A_d.T + GQG     # [1, Eq. 12.42]

    return x_est, P_est, K_est
