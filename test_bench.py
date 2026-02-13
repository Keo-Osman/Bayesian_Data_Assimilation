import numpy as np
import sys, argparse
import random
import time


#region Simulation Config
TIME_STEP = 0.005
TIME_PERIOD = 50 # seconds
STEPS = int(TIME_PERIOD / TIME_STEP)
RNG_SEED = 1 # Keep a constant to have RNG the exact same across runs - still random but will be consitent
# RNG_SEED = random.randint(0, 1_000_000) 
rng = np.random.default_rng(RNG_SEED)
#endregion

#region Command Line Arguments and Model Setup
model_list = ["linear", "lorenz-EnKF", "lorenz-EKF"]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Input model name to run e.g. -m lorenz")
parser.add_argument("-l", "--list-models", action="store_true", help="Lists all available models")
parser.add_argument(
    '-R',
    nargs='+',        
    type=float,
)
parser.add_argument(
    '-P',
    nargs='+',        
    type=float,
)
parser.add_argument(
    '-T',
    nargs='+',        
    type=float,
)
parser.add_argument(
    '-E',
    nargs='+',        
    type=float,
)
parser.add_argument(
    '-O',
    nargs='+',        
    type=float,
)
args = parser.parse_args()

if (args.list_models):
    for i, m in enumerate(model_list):
        print(f'{i+1}. {m}')
    sys.exit()




from models import linear_gaussian, lorenz_attractor_EnKF, lorenz_attractor_EKF
match args.model:
    case "lorenz-EnKF":
        model = lorenz_attractor_EnKF.LorenzModel(TIME_STEP, RNG_SEED)
    case "lorenz-EKF":
        model = lorenz_attractor_EKF.LorenzModel(TIME_STEP, RNG_SEED)
    case "linear":
        model = linear_gaussian.LinearGaussianModel(TIME_STEP, RNG_SEED)
NUM_VARIABLES = model.NUM_VARIABLES
if (args.R):
    if(len(args.R) == 1):
        R = args.R[0] * np.eye(NUM_VARIABLES)
    elif(len(args.R) != NUM_VARIABLES):
        print(f'R should have {NUM_VARIABLES} variables!')
        sys.exit()
    else:
        R = np.diag(args.R)
else:
    R = 2 * np.eye(NUM_VARIABLES)
    
if (args.P):
    if(len(args.P) == 1):
        P = args.P[0] * np.eye(NUM_VARIABLES)
    elif(len(args.P) != NUM_VARIABLES):
        print(f'P should have {NUM_VARIABLES} variables!')
        sys.exit()
    else:
        P = np.diag(args.P)
else:
    P = 2 * np.eye(NUM_VARIABLES)

if (args.T):
    if(len(args.T) == 1):
        true_initial = np.full(NUM_VARIABLES, args.T[0])
    elif (len(args.T) != NUM_VARIABLES):
        print(f'T should have {NUM_VARIABLES} variables!')
        sys.exit()
    else:
        true_initial = args.T
else:
    true_initial = model.TRUE_INITIAL

if (args.E):
    if (len(args.E) == 1):
        initial_belief_error = np.full(NUM_VARIABLES, args.E[0])
    elif (len(args.E) != NUM_VARIABLES):
        print(f'initial_belief_error should have {NUM_VARIABLES} variables!')
        sys.exit()
    else:
        initial_belief_error = np.array(args.E)
else:
    initial_belief_error = np.full(NUM_VARIABLES, 0.1)

initial_value = np.zeros_like(true_initial)
for i, val in enumerate(true_initial):
    initial_value[i] = val * (1 + initial_belief_error[i]) 

model.initialise(R, initial_value, P, true_initial)

if (args.O):
    if(len(args.O) == 1):
        obs_freq = np.full(NUM_VARIABLES, args.O[0])
    elif (len(args.O) != NUM_VARIABLES):
        print(f'obs_freq should have {NUM_VARIABLES} variables!')
        sys.exit()
    else:
        obs_freq = np.array(args.O)
else:
    obs_freq = np.full(NUM_VARIABLES, 2.0)

print(f'Running {model.name}')
print(f'R = \n{R} \n\nInitial Belief Error = \n{initial_belief_error}\n\nTrue Initial =\n {true_initial} \n\nobs_freq =\n {obs_freq} \n\nP =\n {P}')


#endregion

#region True Data
start = time.perf_counter()

t = np.arange(STEPS) * TIME_STEP
true_state = model.generate_true_data(STEPS, TIME_STEP, t)

end = time.perf_counter()
print(f'Generating true data took {end - start:.2g}s')
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

from plot import plot_N_variables
title = model.get_title(R, initial_belief_error)
plot_N_variables(mu, obs_array, true_state, t, NUM_VARIABLES, title, model.variable_names)
#endregion