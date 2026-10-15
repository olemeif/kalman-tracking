import numpy as np

from Train_1D_deceleration.kalman_3rd_degree import kalman_3rd_degree
from simulation import simulate_ground_truth
from tests.filter_checks import DT_S, SIGMA_M, V_TRUE, DECELERATION, assert_anees_consistent


def test_first_step_matches_hand_computation():
    """
    Compares gain, estimate and covariance after the first step with the book's equations.
    Prior P_hat = diag(100^2, 20^2, 10^2), C = [1, 0, 0] -> K = [P_ss, 0, 0] / (P_ss + R).
    """
    s_measured = np.array([12.0, 70.0])
    x_est, P_est, K_est = kalman_3rd_degree(s_measured, DT_S, SIGMA_M, 40.0, 0.5)

    K_s = 100.0 ** 2 / (100.0 ** 2 + SIGMA_M ** 2)
    np.testing.assert_allclose(K_est[0], [K_s, 0.0, 0.0])
    np.testing.assert_allclose(x_est[0], [s_measured[0], 40.0, 0.0])
    np.testing.assert_allclose(P_est[0], np.diag([(1 - K_s) * 100.0 ** 2, 20.0 ** 2, 10.0 ** 2]))


def test_noise_free_braking_is_tracked():
    """
    Perfect measurements of the braking scenario: after the speed has settled at v_end the filter
    must recover speed and acceleration, even from a wrong speed prior.
    """
    n_steps, _, s_true, v_true, a_true = simulate_ground_truth(4000.0, V_TRUE, 0.0, DT_S, DECELERATION)
    x_est, _, _ = kalman_3rd_degree(s_true, DT_S, SIGMA_M, 40.0, 0.5)

    assert abs(x_est[-1, 1] - v_true[-1]) < 0.1
    assert abs(x_est[-1, 2] - a_true[-1]) < 0.05


def test_anees_consistent_with_matched_model():
    """
    Constant-speed truth, so a = 0 holds exactly and Q = 0 matches the model: the filter must be
    consistent with n = 3 states, N * ANEES ~ chi2(N * n) [1, Eq. 9.13].
    """
    n_runs, n_states = 200, 3
    n_steps, _, s_true, v_true, a_true = simulate_ground_truth(2500.0, V_TRUE, 0.0, DT_S)
    x_true = np.column_stack([s_true, v_true, a_true])

    assert_anees_consistent(
        lambda s_measured: kalman_3rd_degree(s_measured, DT_S, SIGMA_M, 40.0, 0.0),
        s_true, x_true, n_steps, n_runs, n_states, 0.99
    )
