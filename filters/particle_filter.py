from distributions import *

def propagate(distribution: ParticleDistribution, transition, Q: np.ndarray, rng: np.random.Generator):
    for i, particle in enumerate(distribution.particles):
        distribution.particles[i] = transition(particle) + rng.multivariate_normal(np.zeros(len(particle)), Q)

def update(distribution: ParticleDistribution, weights: np.ndarray, observation: np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    R_inv = np.linalg.inv(R)
    for i, particle in enumerate(distribution.particles):
        y = observation - H @ particle
        likelihood = np.exp(-0.5 * y.T @ R_inv @ y)
        weights[i] = likelihood
    weights /= weights.sum()

def resample(distribution: ParticleDistribution, weights: np.ndarray, rng: np.random.Generator, spread: np.ndarray, num_variables: int):
    N = distribution.num_particles
    idx = rng.choice(N, size=N, p=weights)
    distribution.particles = distribution.particles[idx] + rng.multivariate_normal(np.zeros(num_variables), spread)
    weights = np.ones(N) / N