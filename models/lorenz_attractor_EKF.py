from models.abstract_model import *
from distributions import *
import filters.extended_kalman_filter as EKF 

class LorenzModel(Model):
    def model_step(self):
        o, r, b = self.parameters
        x, y, z = self.distribution.mean
        Jacobian = np.array([
            [-o, o, 0],
            [r-z, -1, -x],
            [y, x, -b]
        ])

        J = np.eye(self.NUM_VARIABLES) + self.dt * Jacobian

        def func(mean: np.ndarray) -> np.ndarray:
            x, y, z = mean
            ds = np.array([
                o*(y - x),
                x*(r-z) - y,
                x*y - b*z
            ]) * self.dt
            res = mean + ds
            return res

        EKF.propagate(self.distribution, func, J, self.Q)

    def __init__(self, timestep: float, rng_seed: int):
        self.dt = timestep
        self.NUM_VARIABLES = 3
        self.rng = np.random.default_rng(rng_seed)
        
        self.parameters = [10, 28, 8/3]
        self.Q = 1e-9 * np.eye(self.NUM_VARIABLES) * self.dt
        self.TRUE_INITIAL = np.array([1.2, -3 , 4.0]) # Default true value, may be overriden by cmdline arguments in initialise()
        
    def initialise(self, R: np.ndarray, initial_value: np.ndarray, initial_covariance: np.ndarray, true_initial: np.ndarray):
        self.R = R
        mu0 = initial_value
        P0 = initial_covariance
        self.TRUE_INITIAL = true_initial

        self.distribution = Gaussian(mu0, P0)

    def on_observation(self, observation, observed_idx):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]

        EKF.update(self.distribution, observation, H, R_k)

    def generate_true_data(self, STEPS: int, TIME_STEP: float, t: np.ndarray) -> np.ndarray:
        true_state = np.zeros((STEPS, self.NUM_VARIABLES))
        true_state[0] = self.TRUE_INITIAL

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
        return f'Lorenz Extended Kalman Filter (EKF) (R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'
    
    @property
    def variable_names(self) -> List[str]:
        return ["X", "Y", "Z"]
    
    @property
    def name(self) -> str:
        return "Lorenz Attractor with Extended Kalman Filter"