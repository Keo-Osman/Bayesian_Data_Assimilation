from models.abstract_model import *
from distributions import *
import filters.particle_filter as PF 
from scipy.integrate import DOP853, solve_ivp

class LorenzModel(Model):
    def model_step(self):
        o, r, b = self.parameters
        def lorenz_vector(t, particles):
            x = particles[0::3]
            y = particles[1::3]
            z = particles[2::3]
            dx = o*(y-x)
            dy = r*x - y - x*z
            dz = x*y - b*z
            dS = np.empty_like(particles)
            dS[0::3] = dx
            dS[1::3] = dy
            dS[2::3] = dz
            return dS
        
        t0 = self.time
        tf = self.time + self.dt
        def transition(particles: np.ndarray):
            particles_flat = particles.flatten()
            sol = solve_ivp(lorenz_vector, (t0, tf), particles_flat, method='DOP853', t_eval=[tf])
            return sol.y.T.reshape(-1, self.NUM_VARIABLES)
            

        PF.propagate_vectorised(self.distribution, transition, self.Q, self.rng)
        self.time = tf


    def on_observation(self, observation, observed_idx):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]
        PF.update(self.distribution, self.weights, observation, H, R_k, self.rng)
        PF.resample(self.distribution, self.weights, self.rng, 1e-1 * np.eye(self.NUM_VARIABLES), self.NUM_VARIABLES)

    def __init__(self, rng_seed: int):
        self.NUM_VARIABLES = 3
        self.rng = np.random.default_rng(rng_seed)
        self.time = 0

        self.parameters = [10, 28, 8/3]
        self.NUM_PARTICLES = 1000
        self.TRUE_INITIAL = np.array([1.2, -3 , 4.0]) # Default true value, may be overriden by cmdline arguments in initialise()
        
    def initialise(self, Q: np.ndarray, R: np.ndarray, initial_value: np.ndarray, initial_covariance: np.ndarray, true_initial: np.ndarray, timestep: float):
        self.dt = timestep
        self.R = R
        self.Q = Q * self.dt

        self.TRUE_INITIAL = true_initial
        mu = initial_value
        P = initial_covariance

        initial_particles = self.rng.multivariate_normal(mu, P, self.NUM_PARTICLES)
        self.distribution = ParticleDistribution(initial_particles)
        self.weights = np.ones(self.NUM_PARTICLES) / self.NUM_PARTICLES

    def generate_true_data(self, STEPS: int, TIME_STEP: float, t_arr: np.ndarray) -> np.ndarray:
        o, r, b = self.parameters
        def dx(t, state):
            x, y, z = state
            return np.array([
                o*(y - x),
                x*(r-z) - y,
                x*y - b*z
            ])
        time = STEPS * TIME_STEP
        return solve_ivp(dx, (0, time), self.TRUE_INITIAL, method='DOP853', t_eval=t_arr).y.T

    def get_title(self, OBS_VARIANCE: np.ndarray, initial_belief_error: np.ndarray) -> str:
        return f'Lorenz PF (Particle No. = {self.NUM_PARTICLES} R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'
    
    @property
    def variable_names(self) -> List[str]:
        return ["X", "Y", "Z"]
    
    @property
    def name(self) -> str:
        return "Lorenz Attractor with Particle Filter"
    
    @property
    def data_path(self) -> str:
        return "data/true_state/lorenz"