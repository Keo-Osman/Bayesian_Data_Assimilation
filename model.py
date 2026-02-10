# This python script takes in a model and runs relevant simulations using it.
from abc import ABC, abstractmethod
import numpy as np
from distrubutions import Distrubution
from typing import List

class Model(ABC):
    @abstractmethod
    def model_step(self) -> Distrubution:
        pass
    
    @abstractmethod
    # Will always be called after the model step has been done.
    def on_observation(self, observation: np.ndarray, observed_idx: List[int]) -> Distrubution:
        pass
    

    @property
    def mean(self):
        pass

    @property
    def covariance(self):
        pass





