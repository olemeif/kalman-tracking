import numpy as np

from Train_1D_deceleration.kalman_3rd_degree import rose_filter
from simulation import simulate_ground_truth
from tests.filter_checks import DT_S, SIGMA_M, V_TRUE, DECELERATION, assert_anees_consistent


Q_MIN = 1.0e-6
Q_MAX = 1.0
ALPHA_M = 1.0


def test_first_step_matches_hand_computation():
    """
    Compares gain, estimate and covariance after the first step with the book's equations.
    No noise estimation at k = 0, so it equals the 3rd degree Kalman filter:
    Prior P_hat = diag(100^2, 20^2, 10^2), C = [1, 0, 0] -> K = [P_ss, 0, 0] / (P_ss + R).
    """
    s_measured = np.array([12.0, 70.0])
    x_est, P_est, K_est, Q_est = rose_filter(s_measured, DT_S, SIGMA_M, 40.0, Q_MIN, Q_MAX, ALPHA_M)

    K_s = 100.0 ** 2 / (100.0 ** 2 + SIGMA_M ** 2)
    np.testing.assert_allclose(K_est[0], [K_s, 0.0, 0.0])
    np.testing.assert_allclose(x_est[0], [s_measured[0], 40.0, 0.0])
    np.testing.assert_allclose(P_est[0], np.diag([(1 - K_s) * 100.0 ** 2, 20.0 ** 2, 10.0 ** 2]))
    assert np.isnan(Q_est[0])


def test_noise_free_braking_is_tracked():
    """
    Perfect measurements of the braking scenario: after the speed has settled at v_end the filter
    must recover speed and acceleration, even from a wrong speed prior.
    """
    n_steps, _, s_true, v_true, a_true = simulate_ground_truth(4000.0, V_TRUE, 0.0, DT_S, DECELERATION)
    x_est, _, _, _ = rose_filter(s_true, DT_S, SIGMA_M, 40.0, Q_MIN, Q_MAX, ALPHA_M)

    assert abs(x_est[-1, 1] - v_true[-1]) < 0.1
    assert abs(x_est[-1, 2] - a_true[-1]) < 0.05


def test_anees_consistent_with_matched_model():
    """
    Constant-speed truth, so a = 0 holds exactly. Q_min = Q_max = 0 clamps the estimated Q to the
    matched model: the filter must be consistent with n = 3 states, N * ANEES ~ chi2(N * n) [1, Eq. 9.13].
    """
    n_runs, n_states = 200, 3
    n_steps, _, s_true, v_true, a_true = simulate_ground_truth(2500.0, V_TRUE, 0.0, DT_S)
    x_true = np.column_stack([s_true, v_true, a_true])

    assert_anees_consistent(
        lambda s_measured: rose_filter(s_measured, DT_S, SIGMA_M, 40.0, 0.0, 0.0, ALPHA_M)[:3],
        s_true, x_true, n_steps, n_runs, n_states, 0.99
    )
