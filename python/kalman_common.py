import numpy as np

def allocate_storage(n_steps: int, n_states: int):
    """
    Storage for the corrected estimates x_tilde(k), their covariances and the gains.
    The full covariance is kept, because NEES needs the off-diagonal elements.
    """
    x_est = np.zeros((n_steps, n_states))
    P_est = np.zeros((n_steps, n_states, n_states))
    K_est = np.zeros((n_steps, n_states))
    return x_est, P_est, K_est

def correct(x_hat, P_hat, C, R, y):
    """
    Correction step of the Kalman filter, [1, Eq. 12.24 - 12.26] and [1, Eq. 12.38 - 12.40].
    """
    K = P_hat @ C.T @ np.linalg.inv(C @ P_hat @ C.T + R)    # [1, Eq. 12.24 / 12.38]
    x_tilde = x_hat + K @ (y - C @ x_hat)                   # [1, Eq. 12.25 / 12.39]
    P_tilde = (np.eye(len(x_hat)) - K @ C) @ P_hat          # [1, Eq. 12.26 / 12.40]
    return x_tilde, P_tilde, K

def predict(x_tilde, P_tilde, A_d, GQG):
    """
    Prediction step of the Kalman filter, [1, Eq. 12.27 - 12.28] and [1, Eq. 12.41 - 12.42].
    """
    x_hat = A_d @ x_tilde                   # [1, Eq. 12.27 / 12.41]
    P_hat = A_d @ P_tilde @ A_d.T + GQG     # [1, Eq. 12.28 / 12.42]
    return x_hat, P_hat

def kalman_filter(
    s_measured: np.ndarray,
    n_steps: int,
    A_d: np.ndarray,
    C: np.ndarray,
    R: float,
    GQG: np.ndarray,
    x_hat: np.ndarray,
    P_hat: np.ndarray
):
    """
    Runs the linear Kalman filter with constant system matrices over all measurements.
    Args:
        s_measured: Measured position (1D array of length n_steps)
        n_steps: Number of time steps
        A_d: Discrete system matrix
        C: Output matrix
        R: Measurement noise variance
        GQG: Process noise covariance G_d * Q * G_d^T
        x_hat: Initial prior state for time k = 0 (column vector)
        P_hat: Initial prior covariance for time k = 0
    Returns:
        x_est, P_est, K_est as described by the calling filter
    """
    x_est, P_est, K_est = allocate_storage(n_steps, len(x_hat))

    for k in range(n_steps):
        # Correction
        x_tilde, P_tilde, K = correct(x_hat, P_hat, C, R, s_measured[k])

        # Store the corrected estimate for time k
        x_est[k] = x_tilde[:, 0]
        P_est[k] = P_tilde
        K_est[k] = K[:, 0]

        # Prediction for time k+1
        x_hat, P_hat = predict(x_tilde, P_tilde, A_d, GQG)

    return x_est, P_est, K_est
