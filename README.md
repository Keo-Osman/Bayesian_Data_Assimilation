This is a quick guide on how to make you own model using this code.

To make a model make a new python file in the models folder and start with the following boiler plate code

```python
from models.abstract_model import *
from distributions import *

class ModelName(Model):
    def model_step(self):
        # propagate step...
        pass
    
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]
        
        #... update step
    
    def __init__(self, rng: np.random.Generator):
        self.NUM_VARIABLES = #...
        self.rng = rng
        self.TRUE_INITIAL = np.array(#...) # Default true value, may be overriden by cmdline arguments in initialise()

    def initialise(self, Q: np.ndarray, R: np.ndarray, initial_value: np.ndarray, initial_covariance: np.ndarray, true_initial: np.ndarray, timestep: float):
        self.R = R
        self.TRUE_INITIAL = true_initial
        mu = initial_value
        P = initial_covariance
        self.dt = timestep
        self.Q = Q * self.dt
        self.distrubution = #...
    
    def generate_true_data(self, STEPS: int, TIME_STEP: float, t: np.ndarray) -> np.ndarray:
        true_state = np.zeros((STEPS, self.NUM_VARIABLES))

        #... generate true data

        return true_state


    def get_title(self, OBS_VARIANCE: np.ndarray, initial_belief_error: np.ndarray) -> str:
        return f'...'

    @property
    def variable_names(self) -> List[str]:
        return [#...]
    
    @property
    def name(self) -> str:
        return "..."

    @property
    def data_path(self) -> str:
        return "path/to/input-data"
```

Your model should keep track of it's current estimate distributions internally and `model_step()` and `on_observation` should mutate that internally no need to return it.  
You should also implement a way to generate the true synthetic data. In most cases this is just a simple Euler's method or similar for you differential equation. No probability here just use the true initial value.
You should also implement `get_title()` and `variable_names()` this is just for when we plot the data.
Also implement `name()` this is logged to the console and displays the name of the model running.
If your model uses any randomness *like when using ensemble Kalman filter* use `self.rng` for all random generation this allows the same randomness over different runs so you know any changes are the result of the model and not luck. If you want randomness over different runs change the line for RNG_SEED in test_bench.py

When you want to use your model you need to add it into test_bench.py by making the following changes in the Command Line Arguments and Model Setup region:
```python
model_list = [...., MODEL_NAME]
#...
from models import ..., YOUR_MODEL
match args.model:
	#....
    case "YOUR_MODEL":
        model = YOUR_MODEL.YOUR_MODEL(TIME_STEP, RNG_SEED)
```

Then run you model with `python test_bench.py -m MODEL_NAME` it should work from hopefully there with no issue. 
You can change some configurations in the observations region.
You also have the following *optional* command line options
`-T` for true initial
`-E` for initial belief error (proportion not percent e.g. 1.0 not 100%) can be negative
`-O` for observation frequency
`-R` for measurement covariance
`-P` for model initial covariance
You can either put the array full array for this with space separated numbers e.g `-R 1.0 0.1 -2.0` or you can use a single value which will be used for the whole array/covariance e.g. `-R 5.0`.
*Note that for the covariance arguments the array is the leading diagonal, off diagonal elements will always be 0*
You can pass any combination of them in in any order, if you don't pass a value in it will be initialised to some default value that is consistent across runs


If you want to make your own distribution add it into `distributions.py` it can have anything as long as implement a mean and covariance property. This is read only and used when plotting actual estimates you don't have to actually keep track of a mean and covariance. See `ParticleDistribution` for an example of what I mean here.

If you want to make your own filter it should have an `propagate()` and `update()` method. Each of them should take in a specific distribution and mutate it accordingly.