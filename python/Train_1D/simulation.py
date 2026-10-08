import numpy as np

def simulate_ground_truth(
    track_length_m: float,
    v_0: float,
    s_0: float,
    dt_s: float
):
    """
    Simulate the ground truth data for the 1D train scenario.
    Args:
        track_length_m: float, length of the track in meters
        v_0: float, initial velocity of the train in m/s
        s_0: float, initial position of the train in meters
        dt_s: float, time step in seconds
    Returns:
        n_steps: int, number of time steps
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector
        v_true: np.ndarray, true velocity vector
    """

    n_steps = int(track_length_m / (v_0 * dt_s)) + 1
    t = np.arange(n_steps) * dt_s

    s_true = s_0 + v_0 * t
    v_true = np.full(n_steps, v_0)

    return n_steps, t, s_true, v_true

def generate_measurement(
        s_true: np.ndarray,
        sigma_m: float,
        dt_s: float,
        n_steps: int,
        seed: int = None):
    """
    Generate noisy measurements from the ground truth data by adding Gaussian noise.
    Args:
        s_true: np.ndarray, true position vector
        sigma_m: float, standard deviation of the measurement noise
        dt_s: float, time step in seconds
        n_steps: int, number of time steps
        seed: int, random seed for reproducibility
    Returns:
        s_measured: np.ndarray, measured position vector with noise
        v_diff: np.ndarray, estimated velocity vector using finite difference
    """
    rng = np.random.default_rng(seed)
    gps_noise = rng.normal(0, sigma_m, size=n_steps)
    s_measured = s_true + gps_noise

    print(f"Noise: mean = {np.mean(gps_noise):.4f} m, std = {np.std(gps_noise):.4f} m")

    # Speed estimation using finite difference
    v_diff = np.full(n_steps, np.nan)
    v_diff[1:] = np.diff(s_measured) / dt_s

    return s_measured, v_diff

