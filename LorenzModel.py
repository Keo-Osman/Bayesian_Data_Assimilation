from model import *
from distrubutions import *
import ExtendedKalmanFilter as EnKF 

class LorenzModel:
    def __init__(self, timestep: float, rng_seed: int):
        self.dt = timestep
        self.NUM_VARIABLES = 3
        self.rng = np.random.default_rng(rng_seed)
        
        self.parameters = [10, 28, 8/3]
        self.R = 10 * np.eye(self.NUM_VARIABLES)

        mu0 = np.array([1.1, 1.8, 6]) # initial belief_mean
        P0 = 5 * np.eye(self.NUM_VARIABLES)  # initial belief covariance

        self.NUM_PARTICLES = 50
        initial_particles = self.rng.multivariate_normal(mu0, P0, self.NUM_PARTICLES)
        self.distrubution = ParticleDistrubution(initial_particles)

    # state_prev is an array of particles, each particle is an array of state variables
    def model_step(self):
        o, r, b = self.parameters

        def func(particle: np.ndarray) -> np.ndarray:
            x, y, z = particle
            ds = np.array([
                o*(y - x),
                x*(r-z) - y,
                x*y - b*z
            ]) * self.dt
            res = particle + ds
            return res
        
        EnKF.propagate(self.distrubution, func)

        

    def on_observation(self, observation, observed_idx):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]

        EnKF.update(self.distrubution, observation, H, R_k, self.rng)
