from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    # @abstractmethod
    # def sample(self, rng: np.random.Generator) -> np.ndarray:
    #     pass
    @property
    @abstractmethod
    def mean() -> np.ndarray:
        pass

    @property
    @abstractmethod
    def covariance() -> np.ndarray:
        pass

class Gaussian(Distribution):
    def __init__(self, mean: np.ndarray, covariance):
        self._mean = mean
        self._covariance = covariance
    
    # def sample(self, rng: np.random.Generator) -> np.ndarray:
    #     return rng.multivariate_normal(self.mean, self.covariance)
    @property
    def mean(self) -> np.ndarray:
        return self._mean
    
    @property
    def covariance(self) -> np.ndarray:
        return self._covariance
    


class ParticleDistribution(Distribution):
    def __init__(self, particles: np.ndarray):
        # Particles is an array of shape (NUM_PARTICLES, NUM_VARIABLES)
        self.particles = particles
    
    @property
    def mean(self) -> np.ndarray:
        return np.average(self.particles, axis=0)
    
    @property
    def covariance(self) -> np.ndarray:
        return np.cov(self.particles, rowvar=False, bias=False)
    
    @property
    def num_particles(self) -> int:
        return len(self.particles)