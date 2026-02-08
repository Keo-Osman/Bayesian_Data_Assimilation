# This python script takes in a model and runs relevant simulations using it.
from abc import ABC, abstractmethod
import numpy as np
from distrubutions import Distrubution
from typing import List

class Model(ABC):
    @abstractmethod
    def model_step(self, state_prev: Distrubution, dt: float) -> Distrubution:
        pass
    
    @abstractmethod
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]) -> Distrubution:
        pass





