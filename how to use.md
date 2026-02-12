This is a quick guide on how to make you own model using this code.

To make a model make a new python file in the models folder and start with the following boiler plate code

```python
from models.abstract_model import *
from distributions import *

class ModelName(Model):
    def __init__(self, timestep: float, rng: np.random.Generator):
        self.NUM_VARIABLES = #...
        self.TIME_STEP = timestep
        self.rng = rng
        self.distribution = #...

    def model_step(self):
        # propagate step...
        pass
    
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]):
        # Build H and R_k based on observation indices
        H = np.eye(self.NUM_VARIABLES)[observed_idx, :]
        R_k = self.R[np.ix_(observed_idx, observed_idx)]
        
        #... update step
        
    
    def generate_true_data(self, STEPS: int, TIME_STEP: float, t: np.ndarray) -> np.ndarray:
        TRUE_INTITIAL = np.array(#...)
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
You can change some configurations in the Observations region
You can also change R and/or OBS_VARIANCE to change the covariance of the observations. You can also change obs_freq to change how frequently each variable is observed. If you change R, reflect it in you model R.

In the future these should be command line arguments and R will be passed into the models.

If you want to make your own distribution add it into `distributions.py` it can have anything as long as implement a mean and covariance property. This is read only and used when plotting actual estimates you don't have to actually keep track of a mean and covariance. See `ParticleDistribution` for an example of what I mean here.

If you want to make your own filter it should have an `propagate()` and `update()` method. Each of them should take in a specific distribution and mutate it accordingly.