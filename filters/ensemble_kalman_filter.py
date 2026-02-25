from distributions import *

def propagate(distribution: ParticleDistribution, transition, Q: np.ndarray, rng: np.random.Generator):
    for i, particle in enumerate(distribution.particles):
        distribution.particles[i] = transition(particle) + rng.multivariate_normal(np.zeros(len(particle)), Q)

def propagate_vectorised(distrubution: ParticleDistribution, transition, Q: np.ndarray, rng: np.random.Generator):
    distrubution.particles = transition(distrubution.particles) + rng.multivariate_normal(np.zeros(len(Q)), Q, size=distrubution.num_particles)

def update(distribution: ParticleDistribution, observation: np.ndarray, 
           H: np.ndarray, R: np.ndarray, rng: np.random.Generator):
    # flat = distribution.particles.flatten()
    P = distribution.covariance
    S = H @ P @ H.T + R # Innovation vector covariance
    K = np.linalg.solve(S.T, (P @ H.T).T).T # Kalman Gain
    
    N = distribution.num_particles
    z = np.empty((len(observation), N))
    z = observation + rng.multivariate_normal(np.zeros(len(observation)), R, size=N)
    y = z - distribution.particles @ H.T
    distribution.particles += y @ K.T