import numpy as np

def speed_profile(
    t: np.ndarray,
    v_0: float,
    s_0: float,
    deceleration: dict = None
):
    """
    Speed and position for a train at constant speed v_0, with an optional smooth speed change.
    The speed change follows a raised cosine, so the acceleration is continuous (no step at the
    start or end of braking):
        v(t) = v_0 - dv/2 * (1 - cos(pi * tau)),  tau = (t - t_start) / T,  0 <= tau <= 1
    Position is the closed-form integral of v(t).
    Args:
        t: np.ndarray, time vector
        v_0: float, initial velocity of the train in m/s
        s_0: float, initial position of the train in meters
        deceleration: dict with keys t_start_s, duration_s, v_end, or None for constant speed
    Returns:
        s: np.ndarray, position vector
        v: np.ndarray, velocity vector
    """
    if deceleration is None:
        return s_0 + v_0 * t, np.full(t.shape, v_0)

    t_start = deceleration["t_start_s"]
    T = deceleration["duration_s"]
    dv = v_0 - deceleration["v_end"]

    tau = np.clip((t - t_start) / T, 0.0, 1.0)
    v = v_0 - dv / 2 * (1 - np.cos(np.pi * tau))

    # Distance lost against constant speed v_0: integral of dv/2 * (1 - cos(pi * tau)) dt
    # During braking: dv/2 * (T*tau - T/pi * sin(pi*tau)); after braking it stays at dv/2 * T,
    # and the extra distance lost after braking is dv * (t - t_start - T)
    lost = dv / 2 * (T * tau - T / np.pi * np.sin(np.pi * tau)) + dv * np.maximum(t - t_start - T, 0.0)
    s = s_0 + v_0 * t - lost

    return s, v

def simulate_ground_truth(
    track_length_m: float,
    v_0: float,
    s_0: float,
    dt_s: float,
    deceleration: dict = None
):
    """
    Simulate the ground truth data for the 1D train scenario.
    The simulation ends when the train reaches the end of the track.
    Args:
        track_length_m: float, length of the track in meters
        v_0: float, initial velocity of the train in m/s
        s_0: float, initial position of the train in meters
        dt_s: float, time step in seconds
        deceleration: dict with keys t_start_s, duration_s, v_end, or None for constant speed
    Returns:
        n_steps: int, number of time steps
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector
        v_true: np.ndarray, true velocity vector
    """
    # The speed never drops below v_min, so this many steps always covers the track
    v_min = v_0 if deceleration is None else min(v_0, deceleration["v_end"])
    n_max = int(track_length_m / (v_min * dt_s)) + 1
    t = np.arange(n_max) * dt_s

    s_true, v_true = speed_profile(t, v_0, s_0, deceleration)

    # Keep only the samples on the track
    n_steps = int(np.sum(s_true - s_0 <= track_length_m))
    t, s_true, v_true = t[:n_steps], s_true[:n_steps], v_true[:n_steps]

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

    # Speed estimation using finite difference
    v_diff = np.full(n_steps, np.nan)
    v_diff[1:] = np.diff(s_measured) / dt_s

    return s_measured, v_diff

def simulate(scenario: dict, seed: int = None):
    """
    Simulate one run of the 1D train scenario: ground truth plus one noise realization.
    The seed is a parameter (not read from the scenario), so Monte Carlo runs can loop over it.
    Args:
        scenario: dict, scenario as read from the scenario JSON file
        seed: int, random seed for the measurement noise
    Returns:
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector
        v_true: np.ndarray, true velocity vector
        s_measured: np.ndarray, measured position vector with noise
        v_diff: np.ndarray, estimated velocity vector using finite difference
    """
    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]

    n_steps, t, s_true, v_true = simulate_ground_truth(
        scenario["track_length_m"], scenario["train"]["v_0"], scenario["train"]["s_0"], dt_s,
        scenario["train"].get("deceleration"))
    s_measured, v_diff = generate_measurement(s_true, sigma_m, dt_s, n_steps, seed)

    return t, s_true, v_true, s_measured, v_diff
