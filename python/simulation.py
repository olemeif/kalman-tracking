import numpy as np

def speed_profile(
    t: np.ndarray,
    v_0: float,
    s_0: float,
    dt_s: float,
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
        dt_s: float, time step in seconds
        deceleration: dict with keys t_start_s, duration_s, v_end, or None for constant speed
    Returns:
        s: np.ndarray, position vector
        v: np.ndarray, velocity vector
        a: np.ndarray, acceleration vector
    """
    if deceleration is None:
        return s_0 + v_0 * t, np.full(t.shape, v_0), np.zeros(t.shape)

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

    a = np.zeros(t.shape)
    braking = (t >= t_start) & (t <= t_start + T)
    a[braking] = -dv / 2 * np.pi / T * np.sin(np.pi * tau[braking])

    return s, v, a

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
        a_true: np.ndarray, true acceleration vector
    """
    # The speed never drops below v_min, so this many steps always covers the track
    v_min = v_0 if deceleration is None else min(v_0, deceleration["v_end"])
    n_max = int(track_length_m / (v_min * dt_s)) + 1
    t = np.arange(n_max) * dt_s

    s_true, v_true, a_true = speed_profile(t, v_0, s_0, dt_s, deceleration)

    # Keep only the samples on the track
    n_steps = int(np.sum(s_true - s_0 <= track_length_m))
    t, s_true, v_true, a_true = t[:n_steps], s_true[:n_steps], v_true[:n_steps], a_true[:n_steps]

    return n_steps, t, s_true, v_true, a_true

def velocity_profile_2d(
    t: np.ndarray,
    v_0: list,
    maneuvers: list
):
    """
    Cartesian velocity [v_x, v_y] of a surface target that starts at constant velocity v_0 and
    changes its velocity vector in a sequence of smooth maneuvers. No heading angles are used.
    Each maneuver blends from the current velocity v_a to v_end = v_b with the same raised cosine
    weight as the 1D speed change, so the acceleration is continuous:
        w = (1 - cos(pi * tau)) / 2,  tau = (t - t_start) / T,  0 <= tau <= 1
        speed(t)     = |v_a| + (|v_b| - |v_a|) * w
        direction(t) = d / |d|,  d = v_a + (v_b - v_a) * w
    For |v_a| = |v_b| the speed stays constant during the turn.
    Args:
        t: np.ndarray, time vector
        v_0: [v_x, v_y], initial velocity in m/s
        maneuvers: list of dicts with keys t_start_s, duration_s, v_end ([v_x, v_y] in m/s),
            sorted by t_start_s and not overlapping
    Returns:
        v: np.ndarray, velocity (n, 2)
    """
    v_a = np.asarray(v_0, dtype=float)
    v = np.tile(v_a, (len(t), 1))
    t_free = -np.inf   # end of the previous maneuver

    for m in maneuvers:
        v_b = np.asarray(m["v_end"], dtype=float)
        if m["t_start_s"] < t_free:
            raise ValueError("maneuvers must be sorted by t_start_s and must not overlap")
        # d passes through zero if v_b points opposite to v_a: the direction is undefined there
        if v_a[0] * v_b[1] - v_a[1] * v_b[0] == 0 and v_a @ v_b <= 0:
            raise ValueError(f"v_end {m['v_end']} is zero or opposite to the current velocity")

        tau = np.clip((t - m["t_start_s"]) / m["duration_s"], 0.0, 1.0)
        w = (0.5 * (1 - np.cos(np.pi * tau)))[:, None]
        d = v_a + (v_b - v_a) * w
        speed = np.linalg.norm(v_a) + (np.linalg.norm(v_b) - np.linalg.norm(v_a)) * w

        # From t_start on, this maneuver (and its end velocity) overrides the previous profile
        active = t >= m["t_start_s"]
        v[active] = (speed * d / np.linalg.norm(d, axis=1, keepdims=True))[active]
        v_a, t_free = v_b, m["t_start_s"] + m["duration_s"]

    return v

def simulate_ground_truth_2d(
    duration_s: float,
    p_0: list,
    v_0: list,
    dt_s: float,
    maneuvers: list = None,
    n_sub: int = 100
):
    """
    Simulate the ground truth data for a 2D surface target in cartesian coordinates [x, y].
    The position during a constant-speed turn has no closed form, so the velocity is integrated
    on a grid n_sub times finer than dt_s (trapezoidal rule) and sampled every dt_s.
    Args:
        duration_s: float, simulated time in seconds
        p_0: [x, y], initial position in meters
        v_0: [v_x, v_y], initial velocity in m/s
        dt_s: float, time step in seconds
        maneuvers: list of dicts with keys t_start_s, duration_s, v_end, or None for constant velocity
        n_sub: int, integration substeps per time step
    Returns:
        n_steps: int, number of time steps
        t: np.ndarray, time vector (n_steps,)
        s_true: np.ndarray, true position [x, y] (n_steps, 2)
        v_true: np.ndarray, true velocity [v_x, v_y] (n_steps, 2)
        a_true: np.ndarray, true acceleration [a_x, a_y] (n_steps, 2)
    """
    n_steps = int(round(duration_s / dt_s)) + 1
    t = np.arange(n_steps) * dt_s

    h = dt_s / n_sub
    t_fine = np.arange((n_steps - 1) * n_sub + 1) * h
    v_fine = velocity_profile_2d(t_fine, v_0, maneuvers or [])
    s_fine = np.asarray(p_0, dtype=float) + np.vstack(
        [np.zeros(2), np.cumsum((v_fine[1:] + v_fine[:-1]) / 2 * h, axis=0)])
    a_fine = np.gradient(v_fine, h, axis=0)

    return n_steps, t, s_fine[::n_sub], v_fine[::n_sub], a_fine[::n_sub]

def generate_measurement(
        s_true: np.ndarray,
        sigma_m: float,
        dt_s: float,
        n_steps: int,
        seed: int = None):
    """
    Generate noisy measurements from the ground truth data by adding Gaussian noise.
    Works for 1D positions (n_steps,) and cartesian 2D positions (n_steps, 2); in 2D each axis
    gets independent noise with the same sigma_m.
    Args:
        s_true: np.ndarray, true position vector (n_steps,) or (n_steps, 2)
        sigma_m: float, standard deviation of the measurement noise
        dt_s: float, time step in seconds
        n_steps: int, number of time steps
        seed: int, random seed for reproducibility
    Returns:
        s_measured: np.ndarray, measured position vector with noise, same shape as s_true
        v_diff: np.ndarray, estimated velocity vector using finite difference, same shape as s_true
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma_m, size=(n_steps,) + s_true.shape[1:])
    s_measured = s_true + noise

    # Speed estimation using finite difference
    v_diff = np.full(s_measured.shape, np.nan)
    v_diff[1:] = np.diff(s_measured, axis=0) / dt_s

    return s_measured, v_diff

def simulate(scenario: dict, seed: int = None, target: str = None):
    """
    Simulate one run of a scenario: ground truth plus one noise realization.
    The seed is a parameter (not read from the scenario), so Monte Carlo runs can loop over it.
    1D train scenarios have a "train" entry. The 2D surface radar scenario holds several targets
    under "targets"; target picks one of them by name.
    Args:
        scenario: dict, scenario as read from the scenario JSON file
        seed: int, random seed for the measurement noise
        target: str, name of the target in a 2D scenario (ignored for 1D)
    Returns:
        1D: arrays of shape (n_steps,). 2D: arrays of shape (n_steps, 2), columns [x, y].
        t: np.ndarray, time vector (n_steps,)
        s_true: np.ndarray, true position vector
        v_true: np.ndarray, true velocity vector
        a_true: np.ndarray, true acceleration vector
        s_measured: np.ndarray, measured position vector with noise
        v_diff: np.ndarray, estimated velocity vector using finite difference
    """
    if "train" in scenario:
        dt_s = scenario["sensors"]["GPS"]["dt_s"]
        sigma_m = scenario["sensors"]["GPS"]["sigma_m"]

        n_steps, t, s_true, v_true, a_true = simulate_ground_truth(
            scenario["track_length_m"],
            scenario["train"]["v_0"],
            scenario["train"]["s_0"],
            dt_s,
            scenario["train"].get("deceleration"))
    else:
        dt_s = scenario["sensors"]["SRA"]["dt_s"]
        sigma_m = scenario["sensors"]["SRA"]["sigma_m"]
        if target not in scenario["targets"]:
            raise ValueError(f"target must be one of {list(scenario['targets'])}, got {target!r}")

        n_steps, t, s_true, v_true, a_true = simulate_ground_truth_2d(
            scenario["duration_s"],
            scenario["targets"][target]["p_0"],
            scenario["targets"][target]["v_0"],
            dt_s,
            scenario["targets"][target].get("maneuvers"))
    s_measured, v_diff = generate_measurement(s_true, sigma_m, dt_s, n_steps, seed)

    return t, s_true, v_true, a_true, s_measured, v_diff
