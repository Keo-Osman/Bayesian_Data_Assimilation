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

    S = H @ P @ H.T + R # Innovation Vector
    
    I = np.eye(S.shape[0])
    K = P @ H.T @ np.linalg.solve(S, I) # Kalman Gain

    y = observation - H @ mu
    distribution._mean = mu + K @ y

    I = np.eye(P.shape[0])
    distribution._covariance = (I - K @ H) @ P

    # # Multiplying the two normals: Prior belief ~ N(mu_pred, P), Likelihood ~ N(H*observations[k], R_k)
    # P_inv = np.linalg.solve(distribution.covariance, np.eye(len(distribution.covariance)))
    # R_inv = np.linalg.solve(R, np.eye(len(R)))
    # # Cov = (P^-1 + H^T*R^-1*H)^-1
    # S = P_inv + H.T @ R_inv @ H
    # distribution._covariance = np.linalg.solve(S, np.eye(len(S)))
    # # mu = Cov * (P^-1*mu_prev + H^T*R^-1*Observation)
    # distribution._mean = distribution.covariance @ (P_inv @ distribution._mean + H.T @ R_inv @ observation)