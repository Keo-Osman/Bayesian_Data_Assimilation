from distributions import *
import filters.linear_kalman_filter as KF

def propagate(distribution: Gaussian, function, J: np.ndarray, Q: np.ndarray):
    distribution._mean = function(distribution.mean)

    # P = JPJ^T + Q
    distribution._covariance = J @ distribution.covariance @ J.T + Q

# Unlike full EKF we assume linear H
def update(distribution: Gaussian, observation: np.ndarray, H: np.ndarray, R: np.ndarray):
    KF.update(distribution, observation, H, R)
