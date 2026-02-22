from distributions import *

def propagate(distribution: Gaussian, F: np.ndarray, Q: np.ndarray):
    # mu = F @ mu
    distribution._mean = F @ distribution._mean
    # cov = FPF^T + Q
    distribution._covariance = (F @ distribution._covariance @ F.T) + Q


def update(distribution: Gaussian, observation : np.ndarray, 
           H: np.ndarray, R: np.ndarray):

    P = distribution.covariance
    mu = distribution.mean

    S = H @ P @ H.T + R # Innovation vector covariance
    K = np.linalg.solve(S.T, (P @ H.T).T).T # Kalman Gain
    # K = P @ H.T @ np.linalg.inv(S) 
    y = observation - H @ mu # Innovation vector
    
    distribution._mean = mu + K @ y
    distribution._covariance = (np.eye(P.shape[0]) - K @ H) @ P

    # Equivalent but more stable form - negligible performance impact.
    # I = np.eye(P.shape[0])
    # distribution._covariance = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T 