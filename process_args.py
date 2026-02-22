import numpy as np
import sys, argparse

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Input model name to run e.g. -m lorenz")
    parser.add_argument("-D", "--true-data", help="Read true_state data from CSV")
    parser.add_argument("-l", "--list-models", action="store_true", help="Lists all available models")
    parser.add_argument(
        '-r',
        nargs='+',        
        type=float,
        help="Observation noise covariance matrix. Can be one value or (NUM_VARIABLES) amount of values."
    )
    parser.add_argument(
        '-p',
        nargs='+',        
        type=float,
        help="Initial belief covariance matrix. Can be one value or (NUM_VARIABLES) amount of values."
    )
    parser.add_argument(
        '-i',
        nargs='+',        
        type=float,
        help="Initial true value. Can be one value or (NUM_VARIABLES) amount of values."
    )
    parser.add_argument(
        '-e',
        nargs='+',        
        type=float,
        help="Initial belief error (proportion). Can be one value or (NUM_VARIABLES) amount of values."
    )
    parser.add_argument(
        '-o',
        nargs='+',        
        type=float,
        help="Observation frequency. Can be one value or (NUM_VARIABLES) amount of values."
    
    )
    parser.add_argument(
        '-q',
        nargs='+',        
        type=float,
        help="Process/model noise covariance matrix. Can be one value or (NUM_VARIABLES) amount of values."
    
    )
    parser.add_argument(
        '-dt',        
        type=float,
        help="Time step"
    )
    parser.add_argument(
        '-t',        
        type=float,
        help="Time period"
    )
    return parser

def get_matrix_arg(arg, NUM_VARIABLES: int, default_value_scalar: float) -> np.ndarray:
    if(not arg):
        return default_value_scalar * np.eye(NUM_VARIABLES)
    
    if(len(arg) == 1):
        return arg[0] * np.eye(NUM_VARIABLES)
    if(len(arg) == NUM_VARIABLES):
        return np.diag(arg)
    print("!!Wrong amount of variables in argument!!")
    sys.exit()

def get_array_arg(arg, NUM_VARIABLES: int, default_value: float) -> np.ndarray:
    if(not arg):
        return
    
    if(len(arg) == 1):
        return np.full(NUM_VARIABLES, arg[0])
    if(len(arg) == NUM_VARIABLES):
        return np.array(arg)

def process_args(args, NUM_VARIABLES, model):
    if(args.true_data and args.i):
            print("Can not have both -t and -T")


    R = get_matrix_arg(args.r, NUM_VARIABLES, 2)
    Q = get_matrix_arg(args.q, NUM_VARIABLES, 1e-9)
    P = get_matrix_arg(args.p, NUM_VARIABLES, 2)
        
    if (args.i):
        if(len(args.i) == 1):
            true_initial = np.full(NUM_VARIABLES, args.i[0])
        elif (len(args.i) != NUM_VARIABLES):
            print(f'T should have {NUM_VARIABLES} variables!')
            sys.exit()
        else:
            true_initial = args.i
    else:
        if(args.true_data):
            true_initial = -1 # Read from CSV Later
        else:
            true_initial = model.TRUE_INITIAL

    initial_belief_error = get_array_arg(args.e, NUM_VARIABLES, 0.1)


    try:
        if(true_initial == -1):
            pass
    except:
        initial_value = np.zeros_like(true_initial)
        for i, val in enumerate(true_initial):
            initial_value[i] = val * (1 + initial_belief_error[i]) 

    obs_freq = get_array_arg(args.o, NUM_VARIABLES, 2.0)

    if(args.dt):
        time_step = float(args.dt)
    else:
        time_step = -1
    
    if(args.t):
        time_period = float(args.t)
    else:
        time_period = -1

    return [Q, R, P, true_initial, initial_belief_error, initial_value, obs_freq, time_step, time_period]