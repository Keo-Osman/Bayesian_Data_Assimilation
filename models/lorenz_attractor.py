from models.abstract_model import *
from distributions import *
import filters.ensemble_kalman_filter as EnKF 

class LorenzModel(Model):
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
        self.distribution = ParticleDistribution(initial_particles)

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
        
        EnKF.propagate(self.distribution, func)

        

    def on_observation(self, observation, observed_idx):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]

        EnKF.update(self.distribution, observation, H, R_k, self.rng)

    def generate_true_data(self, STEPS: int, TIME_STEP: float, t: np.ndarray) -> np.ndarray:
        TRUE_INTITIAL = np.array([1.0, 2.0, 3.0])
        true_state = np.zeros((STEPS, 3))
        true_state[0] = TRUE_INTITIAL

        def lorenz(vec):
            o, r, b = [10, 28, 8/3]
            x, y, z = vec
            ds = np.array([
                o*(y - x),
                x*(r-z) - y,
                x*y - b*z
            ]) * TIME_STEP
            res = vec + ds
            return res
        
        for i in range(1, len(true_state)):
            true_state[i] = lorenz(true_state[i-1])

        return true_state

    def get_title(self, OBS_VARIANCE: np.ndarray, initial_belief_error: np.ndarray) -> str:
        return f'Lorenz EnKF (Particle No. = {self.NUM_PARTICLES} R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'
    
    @property
    def variable_names(self) -> List[str]:
        return ["X", "Y", "Z"]
    
    @property
    def name(self) -> str:
        return "Lorenz Attractor with Ensemble Kalman Filter"