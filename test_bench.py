import numpy as np
import pandas as pd
import sys
import random
import time


#region Simulation Config
TIME_STEP = 0.005
TIME_PERIOD = 35 # seconds
STEPS = int(TIME_PERIOD / TIME_STEP)
RNG_SEED = 1 # Keep a constant to have RNG the exact same across runs - still random but will be consitent
# RNG_SEED = random.randint(0, 1_000_000) 
rng = np.random.default_rng(RNG_SEED)
#endregion

#region Command Line Arguments and Model Setup
model_list = ["linear", "lorenz-EnKF", "lorenz-EKF"]
from process_args import process_args, add_args

parser = add_args()
args = parser.parse_args()

if (args.list_models):
    for i, m in enumerate(model_list):
        print(f'{i+1}. {m}')
    sys.exit()


from models import linear_gaussian, lorenz_attractor_EnKF, lorenz_attractor_EKF
match args.model:
    case "lorenz-EnKF":
        model = lorenz_attractor_EnKF.LorenzModel(RNG_SEED)
    case "lorenz-EKF":
        model = lorenz_attractor_EKF.LorenzModel(RNG_SEED)
    case "linear":
        model = linear_gaussian.LinearGaussianModel(RNG_SEED)
NUM_VARIABLES = model.NUM_VARIABLES



Q, R, P, true_initial, initial_belief_error, initial_value, obs_freq, dt, time_period = process_args(args, NUM_VARIABLES, model)
if (dt != -1):
    TIME_STEP = dt
if(time_period != -1):
    TIME_PERIOD = time_period 
STEPS = int(TIME_PERIOD / TIME_STEP)


#region True Data
start = time.perf_counter()
if (true_initial is -1):
    df = pd.read_csv(f"{model.data_path}/{args.true_data}.csv", header=None)
    
    def get_relevant(x):
        if (int(x[0] / TIME_STEP) == (x[0] / TIME_STEP) and x[0] <= TIME_PERIOD):
            return True
        return False
    

    data = np.array(list(filter(get_relevant, df.to_numpy())))
    true_state = data.T[1:].T
    true_initial = true_state[0]

    initial_value = np.zeros_like(true_initial)
    for i, val in enumerate(true_initial):
        initial_value[i] = val * (1 + initial_belief_error[i]) 


t = np.arange(STEPS) * TIME_STEP
true_state = model.generate_true_data(STEPS, TIME_STEP, t)

end = time.perf_counter()
print(f'Generating true data took {end - start:.2g}s')
#endregion

model.initialise(Q, R, initial_value, P, true_initial, TIME_STEP)

print(f'Running {model.name}')
print(f'R = \n{R} \n\nInitial Belief Error = \n{initial_belief_error}\n\nTrue Initial =\n {true_initial} \n\nobs_freq =\n {obs_freq} \n\nP =\n {P}')
#endregion

#region Observations
start = time.perf_counter()

p_obs = 1.0 - np.exp(-obs_freq * TIME_STEP)

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

end = time.perf_counter()
print(f'Generating observation data took {end - start:.2g}s')
#endregion

#region Initial Beliefs
mu = np.zeros_like(true_state)

mu[0] = model.distribution.mean

P = np.zeros((STEPS, NUM_VARIABLES, NUM_VARIABLES))
P[0] = model.distribution.covariance

initial_belief_error = np.zeros((NUM_VARIABLES))
for i in range(0, NUM_VARIABLES):
    initial_belief_error[i] = round(((mu[0][i] - true_state[0][i]) / true_state[0][i]) * 100, 1)

# Forecast is what the model believes before observation - may be useful to store separately in the future
# We could propagate foward the forcast a few ms to see what we would have predicted had we not made recent observations
# mu_forecast = np.full((STEPS, NUM_VARIABLES), np.nan)
# mu_forecast[0] = mu[0]

# P_forecast = np.full((STEPS, NUM_VARIABLES, NUM_VARIABLES), np.nan)
# # P_forecast[0] = P[0]
#endregion

#region Core Loop
start = time.perf_counter()
for k in range(1, STEPS):
    # Forecast
    model.model_step()
    mu_pred = model.distribution.mean
    P_pred = model.distribution.covariance 


    # Update
    if(observations[k] is not None):
        # Record forecast seperatley to update for timesteps with observations
        # mu_forecast[k] = mu_pred
        # P_forecast[k] = P_pred

        model.on_observation(observations[k], observed_idx_list[k])
        mu[k] = model.distribution.mean
        P[k] = model.distribution.covariance
    else:
        P[k] = P_pred
        mu[k] = mu_pred

end = time.perf_counter()
print(f'Core loop took {end - start:.4g}s to model {TIME_PERIOD}s ({TIME_PERIOD / (end-start):.4g}x realtime) with a {TIME_STEP} time step ({STEPS} total steps), {(end - start)/STEPS:.4g}s per step.')
#endregion

#region Plot
# Turn observations into np array wiht NaN values when no observations
obs_array = np.full((STEPS, NUM_VARIABLES), np.nan)
for k in range(STEPS):
    if observations[k] is not None:
        obs_array[k, observed_idx_list[k]] = observations[k]

# Calculate
absolute_error = mu - true_state
percent_error = 100 * (absolute_error/true_state)

for i, val in enumerate(percent_error):
    for j, err in enumerate(val):
        if(np.abs(err) > 50):
            percent_error[i][j] = np.nan


from plot import plot_N_variables
title = model.get_title(R, initial_belief_error)
plot_N_variables(mu, obs_array, true_state, absolute_error, percent_error, t, NUM_VARIABLES, title, model.variable_names)
#endregion