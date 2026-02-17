import numpy as np
from scipy.integrate import solve_ivp

TIME_STEP = 0.005
TIME_PERIOD = 35 # seconds
STEPS = int(TIME_PERIOD / TIME_STEP)
TRUE_INITIAL = np.array([1.2, -3.0, 4.0])
FILE_PATH = "data/true_state/lorenz/1.csv"

o, r, b = [10, 28, 8/3]
def dx(t, state):
    x, y, z = state
    return np.array([
        o*(y - x),
        x*(r-z) - y,
        x*y - b*z
    ])


t_arr = np.arange(STEPS + 1) * TIME_STEP
res = solve_ivp(dx, (0, TIME_PERIOD), TRUE_INITIAL, method='DOP853', t_eval=t_arr, rtol=2.3e-14, atol=1e-100).y.T


import pandas as pd 
df = pd.DataFrame(np.hstack((np.array([t_arr]).T, res)))
df.to_csv(FILE_PATH, index=False, header=False)