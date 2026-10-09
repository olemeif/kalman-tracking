import numpy as np
from scipy.stats import chi2

from Train_1D.kalman_2nd_degree import kalman_2nd_degree
from simulation import simulate_ground_truth, generate_measurement
from helpers import nees


DT_S = 1.0
SIGMA_M = 5.0
V_TRUE = 55.55


def ground_truth(n_steps, v=V_TRUE, s_0=0.0):
    t = np.arange(n_steps) * DT_S
    return s_0 + v * t, np.full(n_steps, v)


def test_first_step_matches_hand_computation():
    """
    compares the gain K, the estimate and the covariance P after the first step with exact numbers worked out from the book's equations.
    If a NumPy update changes how matrix maths or broadcasting behaves, this fails first.
    """
    # Prior P_hat = diag(100^2, 20^2), C = [1, 0] -> K = [P_ss, P_vs] / (P_ss + R)
    s_measured = np.array([12.0, 70.0])
    v_0 = 40.0
    x_est, P_est, K_est = kalman_2nd_degree(s_measured, DT_S, SIGMA_M, 2, v_0, 0.01)

    R = SIGMA_M ** 2
    K_s = 100.0 ** 2 / (100.0 ** 2 + R)
    np.testing.assert_allclose(K_est[0], [K_s, 0.0])
    # Prior position is the first measurement, so the innovation is zero
    np.testing.assert_allclose(x_est[0], [s_measured[0], v_0])
    np.testing.assert_allclose(P_est[0], np.diag([(1 - K_s) * 100.0 ** 2, 20.0 ** 2]))


def test_noise_free_converges_from_wrong_speed_prior():
    """
    Feeds in perfect measurements and a speed guess of 0 m/s, and checks the estimate reaches the true 55.55 m/s.
    """
    s_true, _ = ground_truth(n_steps=200)
    x_est, _, _ = kalman_2nd_degree(s_true, DT_S, SIGMA_M, 200, 0.0, 0.0)

    assert abs(x_est[-1, 1] - V_TRUE) < 0.05
    assert abs(x_est[-1, 0] - s_true[-1]) < 0.5


def test_anees_consistent_with_matched_model():
    """
    Runs the filter 200 times on noisy data and checks that its reported uncertainty matches its actual error (the chi-squared bounds from Eq. 9.13).
    This one covers the whole filter, plus nees and SciPy.
    """
    # Truth has constant speed, so Q = 0 matches the true model and the filter must be
    # consistent: N * ANEES ~ chi2(N * n) [1, Eq. 9.13]
    n_runs, n_states, confidence = 200, 2, 0.99
    n_steps, _, s_true, v_true = simulate_ground_truth(2500.0, V_TRUE, 0.0, DT_S)
    x_true = np.column_stack([s_true, v_true])

    nees_k = np.zeros((n_runs, n_steps))
    for i in range(n_runs):
        s_measured, _ = generate_measurement(s_true, SIGMA_M, DT_S, n_steps, seed=i)
        x_est, P_est, _ = kalman_2nd_degree(s_measured, DT_S, SIGMA_M, n_steps, 40.0, 0.0)
        nees_k[i] = nees(x_est - x_true, P_est)

    anees_t = np.mean(nees_k, axis=0)
    alpha = 1 - confidence
    r1 = chi2.ppf(alpha / 2, n_runs * n_states) / n_runs
    r2 = chi2.ppf(1 - alpha / 2, n_runs * n_states) / n_runs

    ss = slice(n_steps // 2, None)
    assert r1 <= np.mean(anees_t[ss]) <= r2
