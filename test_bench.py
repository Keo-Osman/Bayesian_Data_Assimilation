import numpy as np
from scipy.linalg import expm 
from distrubutions import *
import random

#region Simulation Config
TIME_STEP = 0.005
TIME_PERIOD = 50 # seconds
STEPS = int(TIME_PERIOD / TIME_STEP)
NUM_VARIABLES = 3
RNG_SEED = 1 # Keep a constant to have RNG the exact same across runs - still random but will be consitent
# RNG_SEED = random.randint(0, 1_000_000) # Use for randomness across runs
rng = np.random.default_rng(RNG_SEED)
#endregion

#region Model setup
from LinearGaussianModel import *
model = LinearGaussianModel(TIME_STEP)
#endregion

#region True Data
TRUE_INTITIAL = np.array([1.0, 1.0, 1.0])
t = np.arange(STEPS) * TIME_STEP
true_state = np.zeros((STEPS, 3))

for i in range(0, len(true_state)):
    true_state[i] = expm(model.A*t[i]) @ TRUE_INTITIAL
#endregion

#region Observations
# Compute per-timestep observation probability. Keeps observation frequency density the same independant of TIME_STEP
obs_freq = np.array([2.0, 2.0, 2.0])  
p_obs = 1.0 - np.exp(-obs_freq * TIME_STEP)

# Observation noise covariance
R = 0.25 * np.eye(NUM_VARIABLES)

observations = [None] * STEPS
# For each observation keeps track of the indices that correspond to which values where observed
observed_idx_list = [None] * STEPS 


for k in range(STEPS):
    observed_mask = rng.random(NUM_VARIABLES) < p_obs
    observed_idx = np.where(observed_mask)[0]

    if observed_idx.size == 0:
        continue  # no observations this step

    obs_values = true_state[k, observed_idx]

    # Observation noise (covariance projected onto observed variables)
    R_k = R[np.ix_(observed_idx, observed_idx)]
    noise = rng.multivariate_normal(np.zeros(len(observed_idx)), R_k)
    obs_values_noisy = obs_values + noise

    observations[k] = obs_values_noisy
    observed_idx_list[k] = observed_idx
#endregion

#region Initial Beliefs
mu = np.zeros_like(true_state)
mu[0] = model.prediction.mean

P = np.zeros((STEPS, NUM_VARIABLES, NUM_VARIABLES))
P[0] = model.prediction.covariance


# Forecast is what the model believes before observation - may be useful to store separately in the future
# We could propagate foward the forcast a few ms to see what we would have predicted had we not made recent observations
# mu_forecast = np.full((STEPS, NUM_VARIABLES), np.nan)
# mu_forecast[0] = mu[0]

# P_forecast = np.full((STEPS, NUM_VARIABLES, NUM_VARIABLES), np.nan)
# # P_forecast[0] = P[0]
#endregion

#region Core Loop
for k in range(1, STEPS):
    # Forecast
    model.model_step(Gaussian(mu[k-1], P[k-1]))
    mu_pred = model.prediction.mean
    P_pred = model.prediction.covariance 


    # Update
    if(observations[k] is not None):
        # Record forecast seperatley to update for timesteps with observations
        # mu_forecast[k] = mu_pred
        # P_forecast[k] = P_pred

        model.on_observation(observations[k], observed_idx_list[k])
        mu[k] = model.prediction.mean
        P[k] = model.prediction.covariance
    else:
        P[k] = P_pred
        mu[k] = mu_pred
#endregion

#region Plot
# Turn observations into np array wiht NaN values when no observations
obs_array = np.full((STEPS, NUM_VARIABLES), np.nan)
for k in range(STEPS):
    if observations[k] is not None:
        obs_array[k, observed_idx_list[k]] = observations[k]

from plot import plot_N_variables
plot_N_variables(mu, obs_array, true_state, t, NUM_VARIABLES, "Oscillator", ["X", "Y", "Z"])
#endregion

