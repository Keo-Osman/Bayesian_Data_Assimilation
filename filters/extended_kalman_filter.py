from distrubutions import *


def propagate(distrubution: ParticleDistrubution, function):
    for i, particle in enumerate(distrubution.particles):
        distrubution.particles[i] = function(particle)


def update(distrubution: ParticleDistrubution, observation: np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    
    cov = distrubution.covariance
    S = H @ cov @ H.T + R
    K = cov @ H.T @ np.linalg.solve(S, np.eye(len(S)))
    # K = cov @ H.T @ np.linalg.pinv(S)

    for i, particle in enumerate(distrubution.particles):
        y = observation + rng.multivariate_normal(np.zeros(len(observation)), R)
        distrubution.particles[i] += K @ (y - H @ particle)