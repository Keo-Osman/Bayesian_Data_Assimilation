import numpy as np
from scipy.linalg import expm 
from distrubutions import *
import random

#region Simulation Config
TIME_STEP = 0.005
TIME_PERIOD = 30 # seconds
STEPS = int(TIME_PERIOD / TIME_STEP)
NUM_VARIABLES = 3
RNG_SEED = 1 # Keep a constant to have RNG the exact same across runs - still random but will be consitent
# RNG_SEED = random.randint(0, 1_000_000) 
rng = np.random.default_rng(RNG_SEED)
#endregion

#region Model Setup
from LorenzModel import *
model = LorenzModel(TIME_STEP, RNG_SEED)
# from LinearGaussianModel import *
# model = LinearGaussianModel(TIME_STEP, RNG_SEED)
#endregion

#region True Data
TRUE_INTITIAL = np.array([1.0, 2.0, 3.0])
t = np.arange(STEPS) * TIME_STEP
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

# TRUE_INTITIAL = np.array([1.0, 1.0, 1.0])
# t = np.arange(STEPS) * TIME_STEP
# true_state = np.zeros((STEPS, 3))

# for i in range(0, len(true_state)):
#     true_state[i] = expm(model.A*t[i]) @ TRUE_INTITIAL
#endregion

#region Observations
# Compute per-timestep observation probability. Keeps observation frequency density the same independant of TIME_STEP
obs_freq = np.array([5.0, 5.0, 5.0])  
p_obs = 1.0 - np.exp(-obs_freq * TIME_STEP)

# Observation noise covariance
OBS_VARIANCE = 10
R = OBS_VARIANCE  * np.eye(NUM_VARIABLES)

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
mu[0] = model.distrubution.mean

P = np.zeros((STEPS, NUM_VARIABLES, NUM_VARIABLES))
P[0] = model.distrubution.covariance

initial_belief_error = np.zeros((NUM_VARIABLES))
for i in range(0, NUM_VARIABLES):
    initial_belief_error[i] = round(abs((mu[0][i] - true_state[0][i]) / true_state[0][i]) * 100, 1)

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
    model.model_step()
    mu_pred = model.distrubution.mean
    P_pred = model.distrubution.covariance 


    # Update
    if(observations[k] is not None):
        # Record forecast seperatley to update for timesteps with observations
        # mu_forecast[k] = mu_pred
        # P_forecast[k] = P_pred

        model.on_observation(observations[k], observed_idx_list[k])
        mu[k] = model.distrubution.mean
        P[k] = model.distrubution.covariance
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
title = f'Lorenz EnKF (Particle No. = {model.NUM_PARTICLES} R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'
# title = f'Linear Oscillator KF (R={OBS_VARIANCE}, Initial Guess Error (%) {initial_belief_error})'
VARIABLE_NAMES = ["X", "Y", "Z"]
plot_N_variables(mu, obs_array, true_state, t, NUM_VARIABLES, title, VARIABLE_NAMES)
#endregion

