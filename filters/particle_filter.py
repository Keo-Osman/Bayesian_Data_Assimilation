from distributions import *

def propagate(distribution: ParticleDistribution, transition, Q: np.ndarray, rng: np.random.Generator):
    for i, particle in enumerate(distribution.particles):
        distribution.particles[i] = transition(particle) + rng.multivariate_normal(np.zeros(len(particle)), Q)

def propagate_vectorised(distrubution: ParticleDistribution, transition, Q: np.ndarray, rng: np.random.Generator):
    distrubution.particles = transition(distrubution.particles) + rng.multivariate_normal(np.zeros(len(Q)), Q, size=distrubution.num_particles)

def update(distribution, weights, observation, H, R, rng):
    particles = distribution.particles

    y = observation - particles @ H.T
    solve = np.linalg.solve(R, y.T).T
    quad = np.sum(y * solve, axis=1)

    log_w = -0.5 * quad
    log_w -= log_w.max()
    weights[:] = np.exp(log_w)
    weights /= weights.sum()

def resample(distribution: ParticleDistribution, weights: np.ndarray, rng: np.random.Generator, spread: np.ndarray, num_variables: int):
    N = distribution.num_particles
    if (1.0 / np.sum(weights**2) < 0.5 * N):
        idx = rng.choice(N, size=N, p=weights)
        distribution.particles = distribution.particles[idx] + rng.multivariate_normal(np.zeros(num_variables), spread, size=N)
        weights[:] = 1.0 / N