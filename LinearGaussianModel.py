from model import *
from distrubutions import *
from scipy.linalg import expm 

# Models a oscilation with 3 variables
# Equation of d/dt(v) = Av
#     [   0,   1,    0]
# A = [-(a+b), 0,    b]
#     [   b,   0, -(b+c)]
# Solution analytically with matrix exponential
# V_t+1 = exp(A*dt) @ V_t
# Observation noise intepreted as normals centered on true value

class LinearGaussianModel(Model):
    def __init__(self, timestep):
        self.NUM_VARIABLES = 3
        self.TIME_STEP = timestep
        
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
        
        self.prediction = Gaussian(mu, P)

    def model_step(self, state_prev: Gaussian) -> Gaussian:
        # Apply transition state matrix
        self.prediction.mean = self.F @ state_prev.mean

        # P = FPF^T + Q 
        self.prediction.covariance = ((self.F @ state_prev.covariance) @ self.F.T) + self.Q
        return self.prediction
    
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]) -> Gaussian:
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]
        
        # Multiplying the two normals: Prior belief ~ N(mu_pred, P), Likelihood ~ N(H*observations[k], R_k)
        P_inv = np.linalg.inv(self.prediction.covariance)
        R_inv = np.linalg.inv(R_k)
        # Cov = (P^-1 + H^T*R^-1*H)^-1
        self.prediction.covariance = np.linalg.inv(P_inv + H.T @ R_inv @ H)
        # mu = Cov * (P^-1*mu_prev + H^T*R^-1*Observation)
        self.prediction.mean = self.prediction.covariance @ (P_inv @ self.prediction.mean + H.T @ R_inv @ observation)
        
        return self.prediction


