from model import *
from distrubutions import *
from scipy.linalg import expm 
import LinearKalmanFilter as KF

# Models a oscilation with 3 variables
# Equation of d/dt(v) = Av
#     [   0,   1,    0]
# A = [-(a+b), 0,    b]
#     [   b,   0, -(b+c)]
# Solution analytically with matrix exponential
# V_t+1 = exp(A*dt) @ V_t
# Observation noise intepreted as normals centered on true value

class LinearGaussianModel(Model):
    def __init__(self, timestep: float, rng: np.random.Generator):
        self.NUM_VARIABLES = 3
        self.TIME_STEP = timestep
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
        
        self.F = expm(self.A * self.TIME_STEP) # Discretise time - State transition matrix
        self.Q = 1e-7 * np.identity(self.NUM_VARIABLES) * self.TIME_STEP # Model noise - covariance matrix
        self.R = 0.25 * np.eye(self.NUM_VARIABLES) # How model thinks observation noise is distrubuted - covariance matrix of a normal

        mu = [-1.0, 0.1, 2.3] # Initial Guess
        P = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ]) # Initial covariance, no assumptions about correlations e.g the off diagonal elements
        
        self.distrubution = Gaussian(mu, P)

    def model_step(self) -> Gaussian:
        KF.propagate(self.distrubution, self.F, self.Q)
    
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]) -> Gaussian:
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]
        
        KF.update(self.distrubution, observation, H, R_k, self.rng)
        
        return self.distrubution


