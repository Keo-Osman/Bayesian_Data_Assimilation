from distributions import *


def propagate(distribution: ParticleDistribution, function):
    for i, particle in enumerate(distribution.particles):
        distribution.particles[i] = function(particle)


def update(distribution: ParticleDistribution, observation: np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    
    cov = distribution.covariance
    S = H @ cov @ H.T + R
    K = cov @ H.T @ np.linalg.solve(S, np.eye(len(S)))
    # K = cov @ H.T @ np.linalg.pinv(S)

    for i, particle in enumerate(distribution.particles):
        z = observation + rng.multivariate_normal(np.zeros(len(observation)), R)
        distribution.particles[i] += K @ (z - H @ particle)