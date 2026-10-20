import numpy as np
from math import sqrt

from kalman_common import allocate_storage, kalman_filter

def system_matrices(dt_s: float):
    """
    System description of the 3rd degree kinematic model, only the position is measured.
    """
    A_d = np.array([[1, dt_s, 0.5*dt_s**2], [0, 1, dt_s], [0, 0, 1]])     # [1, Eq. 12.29]
    C = np.array([[1, 0, 0]])                                             # y(k) = s(k)
    return A_d, C

def process_noise_GG(dt_s: float):
    """
    G_d * G_d^T for Q = 1, so that Q*(k) = G_d * Q(k) * G_d^T = GG * Q(k) [1, Eq. 12.10].
    """
    GG = np.array([[1/4*dt_s**4, 1/2*dt_s**3, 1/2*dt_s**2],
                   [1/2*dt_s**3, dt_s**2,     dt_s],
                   [1/2*dt_s**2, dt_s,        1]])  # method 1: direct discretization, [1, p. 138]
    # Method 2, [1, Section 12.3] (swap in for the active GG):
    # GG = np.array([[1/36*dt_s**4, 1/12*dt_s**3, 1/6*dt_s**2], [1/12*dt_s**3, 1/4*dt_s**2, 1/2*dt_s], [1/6*dt_s**2, 1/2*dt_s, 1]])
    # Method 3, [1, Section 12.3]:
    # GG = np.array([[1/20*dt_s**4, 1/8*dt_s**3, 1/6*dt_s**2], [1/8*dt_s**3, 1/3*dt_s**2, 1/2*dt_s], [1/6*dt_s**2, 1/2*dt_s, 1]])
    return GG

def initial_prior(s_measured: np.ndarray, v_0: float):
    """
    Initial prior for time k = 0. The filter must not use the ground truth.
    Position: rough guess with a large variance, so the first measurement dominates.
    """
    x_hat = np.array([[s_measured[0]], [v_0], [0.0]])
    P_hat = np.array([[100.0 ** 2, 0, 0], [0, 20.0 ** 2, 0], [0, 0, 10.0 ** 2]])
    return x_hat, P_hat

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

    A_d, C = system_matrices(dt_s)
    GQG = process_noise_GG(dt_s) * Q
    x_hat, P_hat = initial_prior(s_measured, v_0)

    return kalman_filter(s_measured, n_steps, A_d, C, R, GQG, x_hat, P_hat)

def rose_filter(
    s_measured: np.ndarray,
    dt_s: float,
    sigma_m: float,
    v_0: float,
    Q_min: float,
    Q_max: float,
    alpha_M: float,
):
    """
    Special version of ROSE filter for the 3rd degree kinematic model (position, speed, acceleration) in 1D.
    Since we already know the standard deviation of the position measurement noise, we will use the ROSE
    filter only to estimate the process noise Q.
    As described in [1, Chapter 6, Eq. 6.15 - 6.22] and [1, Section 16.2].
    Args:
        s_measured: Measured position (1D array of length n_steps)
        dt_s: Time step in seconds
        sigma_m: Standard deviation of the position measurement noise (known, R is kept constant)
        v_0: Initial velocity estimate (used for the initial prior)
        Q_min: Lower limit of the estimated process noise Var(z_a) in (m/s^2)^2.
            Used during steady motion and whenever the estimate is implausible (negative).
        Q_max: Upper limit of the estimated process noise Var(z_a) in (m/s^2)^2.
            Caps Q during maneuvers.
        alpha_M: Gain of the first-order IIR filter that averages the innovation covariance M,
            0 < alpha_M <= 1. The averaging time constant is roughly dt_s / alpha_M.
    Returns:
        x_est: Corrected state estimates x_tilde(k) = [s, v, a] (array of shape (n_steps, 3))
        P_est: Corrected covariances P_tilde(k), full 3x3 matrix (array of shape (n_steps, 3, 3))
        K_est: Kalman gains [K_s, K_v, K_a] (array of shape (n_steps, 3))
        Q_est: Estimated process noise used for the prediction into step k (array of shape (n_steps,),
            NaN for k = 0, where the initial prior is used)
    """
    n_steps = len(s_measured)
    R = sigma_m ** 2  # [1, Eq. 1.18], known and constant

    A_d, C = system_matrices(dt_s)
    GG = process_noise_GG(dt_s)
    # Coefficient of Q in the measured element C*Q*(k)*C^T = dt_s^4/4 * Q, used for the
    # comparison of coefficients in [1, Eq. 6.17], see [1, Section 6.3]
    CGGC = (C @ GG @ C.T).item()
 
    x_hat, P_hat = initial_prior(s_measured, v_0)

    # Averaged innovation covariance M(k) = E(dy * dy^T) [1, Eq. 6.16], initialized small as in
    # [1, Section 11.4] (M = 0) and [1, Section 16.2] (M = C*P_tilde*C^T). Starting with R means
    # "no process noise yet". The large prior C*P_hat*C^T + R would hold Q at Q_max for a long time.
    M = R
 
    x_est, P_est, K_est = allocate_storage(n_steps, 3)
    Q_est = np.full(n_steps, np.nan)
 
    x_tilde = None
    P_tilde = None
 
    for k in range(n_steps):
        y = s_measured[k]
 
        if k > 0:
            # --- Prediction of the state ---
            x_hat = A_d @ x_tilde                                       # [1, Eq. 6.18], B_d*u = 0
 
            # --- Estimation of the system noise ---
            dy = y - (C @ x_hat).item()                                 # [1, Eq. 6.15], D*u = 0
            M = alpha_M * dy**2 + (1 - alpha_M) * M                     # [1, Eq. 6.16] as IIR filter [1, Eq. 11.33]
            # C*Q*(k-1)*C^T = M(k) - R(k) - C*A_d*P_tilde(k-1)*A_d^T*C^T  [1, Eq. 6.17]
            CAPAC = (C @ A_d @ P_tilde @ A_d.T @ C.T).item()
            Q = (M - R - CAPAC) / CGGC                                  # comparison of coefficients [1, Section 6.3]
            Q = min(max(Q, Q_min), Q_max)                               # limit implausible values [1, p. 87, Section 16.2]
            Q_est[k] = Q
 
            # --- Prediction of the covariance ---
            P_hat = A_d @ P_tilde @ A_d.T + GG * Q                      # [1, Eq. 6.19]
 
        # --- Correction ---
        K = P_hat @ C.T / ((C @ P_hat @ C.T).item() + R)               # [1, Eq. 6.20]
        x_tilde = x_hat + K * (y - (C @ x_hat).item())                 # [1, Eq. 6.21]
        P_tilde = (np.eye(3) - K @ C) @ P_hat                          # [1, Eq. 6.22]
 
        # Store the corrected estimate for time k
        x_est[k] = x_tilde[:, 0]
        P_est[k] = P_tilde
        K_est[k] = K[:, 0]
 
    return x_est, P_est, K_est, Q_est