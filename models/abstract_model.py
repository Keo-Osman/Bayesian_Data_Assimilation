# This python script takes in a model and runs relevant simulations using it.
from abc import ABC, abstractmethod
import numpy as np
from typing import List

class Model(ABC):
    @abstractmethod
    def model_step(self):
        pass
    
    @abstractmethod
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]):
        pass

    @abstractmethod
    def get_title(self, OBS_VARIANCE: np.ndarray, initial_belief_error: np.ndarray) -> str:
        pass

    @abstractmethod
    def generate_true_data(self, STEPS: int, TIME_STEP: float, t: np.ndarray) -> np.ndarray:
        pass

    @property
    def variable_names(self) -> List[str]:
        pass

    @property
    def name(self) -> str:
        pass





