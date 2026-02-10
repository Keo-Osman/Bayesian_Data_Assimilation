from distrubutions import *

def propagate(distrubution: Gaussian, F: np.ndarray, Q: np.ndarray):
    # mu = F @ mu
    distrubution._mean = F @ distrubution._mean
    # cov = FPF^T + Q
    distrubution._covariance = (F @ distrubution._covariance @ F.T) + Q


def update(distrubution: Gaussian, observation : np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    
    # Multiplying the two normals: Prior belief ~ N(mu_pred, P), Likelihood ~ N(H*observations[k], R_k)
    P_inv = np.linalg.solve(distrubution.covariance, np.eye(len(distrubution.covariance)))
    R_inv = np.linalg.solve(R, np.eye(len(R)))
    # Cov = (P^-1 + H^T*R^-1*H)^-1
    S = P_inv + H.T @ R_inv @ H
    distrubution._covariance = np.linalg.solve(S, np.eye(len(S)))
    # mu = Cov * (P^-1*mu_prev + H^T*R^-1*Observation)
    distrubution._mean = distrubution.covariance @ (P_inv @ distrubution._mean + H.T @ R_inv @ observation)