from models.abstract_model import *
from distributions import *
import filters.extended_kalman_filter as EKF 
from scipy.integrate import solve_ivp, DOP853 

class LorenzModel(Model):
    def model_step(self):
        o, r, b = self.parameters

        def dx(t, state):
            x, y, z = state
            return np.array([
                o*(y - x),
                x*(r-z) - y,
                x*y - b*z
            ])
        
        t0 = self.time
        tf = self.time + self.dt

        def transition(particle: np.ndarray) -> np.ndarray:
            solver = DOP853(dx, t0, particle, tf, rtol=1e-10, atol=1e-12)
            while solver.status == 'running':
                solver.step()
            return solver.y

        x, y, z = self.distribution.mean
        Jacobian = np.array([
            [-o, o, 0],
            [r-z, -1, -x],
            [y, x, -b]
        ])

        J = np.eye(self.NUM_VARIABLES) + self.dt * Jacobian
        
        EKF.propagate(self.distribution, transition, J, self.Q)
        self.time =tf

    def __init__(self, rng_seed: int):
        self.time = 0
        self.NUM_VARIABLES = 3
        self.rng = np.random.default_rng(rng_seed)
        
        self.parameters = [10, 28, 8/3]
        self.TRUE_INITIAL = np.array([1.2, -3 , 4.0]) # Default true value, may be overriden by cmdline arguments in initialise()
        
        
    def initialise(self, Q: np.ndarray, R: np.ndarray, initial_value: np.ndarray, initial_covariance: np.ndarray, true_initial: np.ndarray, timestep: float):
        self.dt = timestep
        self.R = R
        self.Q = Q * self.dt

        self.TRUE_INITIAL = true_initial
        mu = initial_value
        P = initial_covariance

        self.Q = Q * self.dt
        self.distribution = Gaussian(mu, P)

    def on_observation(self, observation, observed_idx):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]

        EKF.update(self.distribution, observation, H, R_k)
        o, r, b = self.parameters
        def dx(t, state):
            x, y, z = state
            return np.array([
                o*(y - x),
                x*(r-z) - y,
                x*y - b*z
            ])
        self.solver = DOP853(dx, self.time, self.distribution.mean, np.inf, rtol=1e-10, atol=1e-12)

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
        return f'Lorenz Extended Kalman Filter (EKF) (R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'

    @property
    def variable_names(self) -> List[str]:
        return ["X", "Y", "Z"]
    
    @property
    def name(self) -> str:
        return "Lorenz Attractor with Extended Kalman Filter"
    
    @property
    def data_path(self) -> str:
        return "data/true_state/lorenz"