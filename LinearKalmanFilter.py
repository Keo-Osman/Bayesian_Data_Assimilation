from distrubutions import *

def propagate(distrubution: Gaussian, F: np.ndarray, Q: np.ndarray):
    distrubution._mean = F @ distrubution._mean
    distrubution._covariance = (F @ distrubution._covariance @ F.T) + Q


def update(distrubution: Gaussian, observation : np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    
    # Multiplying the two normals: Prior belief ~ N(mu_pred, P), Likelihood ~ N(H*observations[k], R_k)
    P_inv = np.linalg.inv(distrubution.covariance)
    R_inv = np.linalg.inv(R)
    # Cov = (P^-1 + H^T*R^-1*H)^-1
    distrubution._covariance = np.linalg.inv(P_inv + H.T @ R_inv @ H)
    # mu = Cov * (P^-1*mu_prev + H^T*R^-1*Observation)
    distrubution._mean = distrubution.covariance @ (P_inv @ distrubution._mean + H.T @ R_inv @ observation)