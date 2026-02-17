from models.abstract_model import *
from distributions import *
from scipy.linalg import expm 
import filters.linear_kalman_filter as KF

# Models a oscilation with 3 variables
# Equation of d/dt(v) = Av
#     [   0,   1,    0]
# A = [-(a+b), 0,    b]
#     [   b,   0, -(b+c)]
# Solution analytically with matrix exponential
# V_t+1 = exp(A*dt) @ V_t
# Observation noise intepreted as normals centered on true value

class LinearGaussianModel(Model):
    def model_step(self):
        KF.propagate(self.distribution, self.F, self.Q)
    
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]
        
        KF.update(self.distribution, observation, H, R_k)
        


    def __init__(self, rng: np.random.Generator):
        self.NUM_VARIABLES = 3
        self.rng = rng
        
        # Parameters 
        self.a = 0.5
        self.b = 0.3
        self.c = 0.8
        
        # ODE matrix - d/dt[x] = Ax
        self.A = np.array([
            [0.0, 1.0, 0.0],
            [-(self.a+self.b), 0.0, self.b],
            [self.b, 0.0, -(self.b+self.c)]
        ])
        
        self.F = expm(self.A * self.dt) # Discretise time - State transition matrix

        self.TRUE_INITIAL  = np.array([1.0, -2.3, 5.0]) # Default true value, may be overriden by cmdline arguments in initialise()

    def initialise(self, Q: np.ndarray,  R: np.ndarray, initial_value: np.ndarray, initial_covariance: np.ndarray, true_initial: np.ndarray, timestep: float):
        self.dt = timestep
        self.R = R
        self.TRUE_INITIAL = true_initial
        mu = initial_value
        P = initial_covariance
        self.Q = Q * self.dt
        self.distribution = Gaussian(mu, P)

    def generate_true_data(self, STEPS: int, TIME_STEP: float, t: np.ndarray) -> np.ndarray:
        true_state = np.zeros((STEPS, self.NUM_VARIABLES))

        for i in range(0, len(true_state)):
            true_state[i] = expm(self.A*t[i]) @ self.TRUE_INITIAL

        return true_state


    def get_title(self, OBS_VARIANCE: np.ndarray, initial_belief_error: np.ndarray) -> str:
        return f'Linear Oscillator KF (R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'
    
    @property
    def variable_names(self) -> List[str]:
        return ["X", "Y", "Z"]
    
    
    @property
    def name(self) -> str:
        return "Linear Oscillator with Kalman Filter"

    @property
    def data_path(self) -> str:
        return "data/true_state/linear-oscillator"

