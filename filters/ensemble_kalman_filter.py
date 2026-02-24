from distributions import *

def propagate(distribution: ParticleDistribution, transition, Q: np.ndarray, rng: np.random.Generator):
    for i, particle in enumerate(distribution.particles):
        distribution.particles[i] = transition(particle) + rng.multivariate_normal(np.zeros(len(particle)), Q)


def update(distribution: ParticleDistribution, observation: np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    
    P = distribution.covariance
    S = H @ P @ H.T + R # Innovation vector covariance
    K = np.linalg.solve(S.T, (P @ H.T).T).T # Kalman Gain
    

    for i, particle in enumerate(distribution.particles):
        z = observation + rng.multivariate_normal(np.zeros(len(observation)), R)
        distribution.particles[i] += K @ (z - H @ particle)