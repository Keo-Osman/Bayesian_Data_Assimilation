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

def process_args(args, NUM_VARIABLES, model):
    if(args.true_data and args.i):
            print("Can not have both -t and -T")

    if (args.r):
        if(len(args.r) == 1):
            R = args.r[0] * np.eye(NUM_VARIABLES)
        elif(len(args.r) != NUM_VARIABLES):
            print(f'R should have {NUM_VARIABLES} variables!')
            sys.exit()
        else:
            R = np.diag(args.r)
    else:
        R = 2 * np.eye(NUM_VARIABLES)
    
    if (args.q):
        if(len(args.q) == 1):
            Q = args.q[0] * np.eye(NUM_VARIABLES)
        elif(len(args.q) != NUM_VARIABLES):
            print(f'Q should have {NUM_VARIABLES} variables!')
            sys.exit()
        else:
            Q = np.diag(args.q)
    else:
        Q = 1e-2 * np.eye(NUM_VARIABLES)
        
    if (args.p):
        if(len(args.p) == 1):
            P = args.p[0] * np.eye(NUM_VARIABLES)
        elif(len(args.p) != NUM_VARIABLES):
            print(f'P should have {NUM_VARIABLES} variables!')
            sys.exit()
        else:
            P = np.diag(args.p)
    else:
        P = 2 * np.eye(NUM_VARIABLES)

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

    if (args.e):
        if (len(args.e) == 1):
            initial_belief_error = np.full(NUM_VARIABLES, args.e[0])
        elif (len(args.e) != NUM_VARIABLES):
            print(f'initial_belief_error should have {NUM_VARIABLES} variables!')
            sys.exit()
        else:
            initial_belief_error = np.array(args.e)
    else:
        initial_belief_error = np.full(NUM_VARIABLES, 0.1)

    if(true_initial != -1):
        initial_value = np.zeros_like(true_initial)
        for i, val in enumerate(true_initial):
            initial_value[i] = val * (1 + initial_belief_error[i]) 
    else:
        initial_value = -1

    if (args.o):
        if(len(args.o) == 1):
            obs_freq = np.full(NUM_VARIABLES, args.o[0])
        elif (len(args.o) != NUM_VARIABLES):
            print(f'obs_freq should have {NUM_VARIABLES} variables!')
            sys.exit()
        else:
            obs_freq = np.array(args.o)
    else:
        obs_freq = np.full(NUM_VARIABLES, 2.0)

    if(args.dt):
        time_step = float(args.dt)
    else:
        time_step = -1
    
    if(args.t):
        time_period = float(args.t)
    else:
        time_period = -1

    return [Q, R, P, true_initial, initial_belief_error, initial_value, obs_freq, time_step, time_period]