import numpy as np

from simulation import generate_measurement
from helpers import nees, anees_bounds


DT_S = 1.0
SIGMA_M = 5.0
V_TRUE = 55.55
DECELERATION = {"t_start_s": 30.0, "duration_s": 10.0, "v_end": 40.0}


def assert_anees_consistent(kalman, s_true, x_true, n_steps, n_runs, n_states, confidence):
    """
    Runs the filter n_runs times on noisy measurements of s_true and checks that the ANEES in steady
    state lies inside the chi-squared bounds: N * ANEES ~ chi2(N * n) [1, Eq. 9.13].
    kalman maps the measured positions to (x_est, P_est, K_est).
    """
    nees_k = np.zeros((n_runs, n_steps))
    for i in range(n_runs):
        s_measured, _ = generate_measurement(s_true, SIGMA_M, DT_S, n_steps, seed=i)
        x_est, P_est, _ = kalman(s_measured)
        nees_k[i] = nees(x_est - x_true, P_est)

    r1, r2 = anees_bounds(n_runs, n_states, confidence)
    ss = slice(n_steps // 2, None)
    assert r1 <= np.mean(np.mean(nees_k, axis=0)[ss]) <= r2
