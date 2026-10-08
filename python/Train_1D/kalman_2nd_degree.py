import numpy as np

def kalman_2nd_degree(
    s_measured: np.ndarray,
    dt_s: float,
    sigma_m: float,
    n_steps: int,
    v_0: float
):
    """
    Kalman filter for a 2nd degree kinematic model (position, nearly constant speed) in 1D.
    As described in [1, Section 12.2].
    Args:
        s_measured: Measured position (1D array of length n_steps)
        dt_s: Time step in seconds
        sigma_m: Standard deviation of the position measurement noise
        n_steps: Number of time steps
        v_0: Initial velocity estimate (used for the initial prior)
    Returns:
        s_est: Estimated position (1D array of length n_steps)
        v_est: Estimated velocity (1D array of length n_steps)
        P_s: Estimated position variance (1D array of length n_steps)
        P_v: Estimated velocity variance (1D array of length n_steps)
        K_s: Kalman gain for position (1D array of length n_steps)
        K_v: Kalman gain for velocity (1D array of length n_steps)
    """

    R = sigma_m ** 2  # [1, Eq. 1.18]
    Q = 0.01  # Var(z_v) in (m/s)^2 per step, eyeballed (no model of the train's acceleration)

    A_d = np.array([[1, dt_s], [0, 1]])     # [1, Eq. 12.4]
    C = np.array([[1, 0]])                  # [1, Eq. 12.3]

    # GQG = np.array([[dt_s**2,     dt_s],     [dt_s,     1]]) * Q  # [1, Eq. 12.11]
    GQG   = np.array([[1/4*dt_s**2, 1/2*dt_s], [1/2*dt_s, 1]]) * Q  # [1, Eq. 12.17]
    # GQG = np.array([[1/3*dt_s**2, 1/2*dt_s], [1/2*dt_s, 1]]) * Q  # [1, Eq. 12.19]

    # Initial prior for time k = 0. The filter must not use the ground truth.
    # Position: rough guess with a large variance, so the first measurement dominates.
    x_hat = np.array([[s_measured[0]], [v_0]])
    P_hat = np.array([[100.0 ** 2, 0], [0, 20.0 ** 2]])

    # Storage for the corrected estimates x_tilde(k) and their variances
    s_est = np.zeros(n_steps)   # state estimate vector
    v_est = np.zeros(n_steps)   # velocity estimate vector
    P_s = np.zeros(n_steps)     # Position covariance estimate vector
    P_v = np.zeros(n_steps)     # Velocity covariance estimate vector
    K_s = np.zeros(n_steps)     # Kalman gain, position component
    K_v = np.zeros(n_steps)     # Kalman gain, velocity component

    for k in range(n_steps):
        # Correction
        K = P_hat @ C.T @ np.linalg.inv(C @ P_hat @ C.T + R)    # [1, Eq. 12.24]
        x_tilde = x_hat + K @ (s_measured[k] - C @ x_hat)       # [1, Eq. 12.25]
        P_tilde = (np.eye(2) - K @ C) @ P_hat                   # [1, Eq. 12.26]

        # Store the corrected estimate for time k
        s_est[k] = x_tilde[0, 0]
        v_est[k] = x_tilde[1, 0]
        P_s[k] = P_tilde[0, 0]
        P_v[k] = P_tilde[1, 1]
        K_s[k] = K[0, 0]
        K_v[k] = K[1, 0]

        # Prediction for time k+1
        x_hat = A_d @ x_tilde                   # [1, Eq. 12.27]
        P_hat = A_d @ P_tilde @ A_d.T + GQG     # [1, Eq. 12.28]

    return s_est, v_est, P_s, P_v, K_s, K_v