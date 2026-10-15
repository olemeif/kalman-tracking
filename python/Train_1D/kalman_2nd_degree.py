import numpy as np

from kalman_common import kalman_filter

def kalman_2nd_degree(
    s_measured: np.ndarray,
    dt_s: float,
    sigma_m: float,
    n_steps: int,
    v_0: float,
    Q: float
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
        Q: Process noise Var(z_v) in (m/s)^2 per step
    Returns:
        x_est: Corrected state estimates x_tilde(k) = [s, v] (array of shape (n_steps, 2))
        P_est: Corrected covariances P_tilde(k), full 2x2 matrix (array of shape (n_steps, 2, 2))
        K_est: Kalman gains [K_s, K_v] (array of shape (n_steps, 2))
    """

    R = sigma_m ** 2  # [1, Eq. 1.18]

    A_d = np.array([[1, dt_s], [0, 1]])     # [1, Eq. 12.4]
    C = np.array([[1, 0]])                  # [1, Eq. 12.3]

    # GQG = np.array([[dt_s**2,     dt_s],     [dt_s,     1]]) * Q  # [1, Eq. 12.11]
    GQG   = np.array([[1/4*dt_s**2, 1/2*dt_s], [1/2*dt_s, 1]]) * Q  # [1, Eq. 12.17]
    # GQG = np.array([[1/3*dt_s**2, 1/2*dt_s], [1/2*dt_s, 1]]) * Q  # [1, Eq. 12.19]

    # Initial prior for time k = 0. The filter must not use the ground truth.
    # Position: rough guess with a large variance, so the first measurement dominates.
    x_hat = np.array([[s_measured[0]], [v_0]])
    P_hat = np.array([[100.0 ** 2, 0], [0, 20.0 ** 2]])

    return kalman_filter(s_measured, n_steps, A_d, C, R, GQG, x_hat, P_hat)
