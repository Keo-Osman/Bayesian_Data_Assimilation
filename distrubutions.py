from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class DistrubutionType(Enum):
    GAUSSIAN = 1

class Distrubution(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        pass

class Gaussian(Distrubution):
    def __init__(self, mean: np.ndarray, covariance):
        self.mean = mean
        self.covariance = covariance
    
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.covariance)