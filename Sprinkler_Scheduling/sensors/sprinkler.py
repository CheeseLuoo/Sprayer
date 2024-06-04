import numpy as np
from typing import List

# from ..utilities import GridMap
from .sensor import ISensor


class Sprinkler(ISensor):
    """Sprinkler sensor and operator."""
    def __init__(
        self,
        Setting,
    ) -> None:
        """

        Parameters
        ----------
        rate : float
            Sensing data update rate.
        env: np.ndarray, shape=(num_rows, num_cols)
            A matrix indicating the ground-truth values at different locations.
        env_extent: List[float], (xmin, xmax, ymin, ymax)
            Environment extent
        noise_scale: float
            Standard deviation of the observational Gaussian white noise.

        """
        super().__init__(Setting.sensing_rate)
        self.env = Setting.env.copy()
        self.noise_scale = Setting.noise_scale

    def sense(
        self,
        states: np.ndarray,
        rng = None,
    ) -> np.ndarray:
        """Get sensor observations.

        Parameters
        ----------
        states : np.ndarray, shape=(num_samples, dim_state)
            Get sensor observatinos at the given states.
        rng : np.random.RandomState, optional
            Random number generator for making the observation noisy.

        Returns
        -------
        observations: np.ndarray, shape=(num_samples, )
            Observation at the given state.

        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        observations = self.env[states[:, 0].astype(int), states[:, 1].astype(int)]
        if rng is not None:
            observations = rng.normal(loc=observations, scale=self.noise_scale)
        return observations
    
    def set_env(
        self,
        env: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        env: np.ndarray, shape=(num_rows, num_cols)
            A matrix indicating the ground-truth values at different locations.

        """
        self.env = env.copy()
        
    def spray(
        self,
        x_spray: np.ndarray,
        spray_flag: bool,
        extent: List[float]
    ) -> None:
        """
        Parameters
        ----------
        env: np.ndarray, shape=(num_rows, num_cols)
            A matrix indicating the ground-truth values at different locations.

        """
        env = self.env.copy()
        effect = 0
        if spray_flag:
            if x_spray.ndim == 1:
                x_spray = x_spray.reshape(1, -1)
            for a in range(3):
                for b in range(3):
                    c1 = x_spray[:,0] - 1 + a
                    c2 = x_spray[:,1] - 1 + b
                    c1 = int(c1)
                    c2 = int(c2)
                    if c1 < extent[0] or c1 >= extent[1] or c2 < extent[2] or c2 >= extent[3]:
                        continue
                    elif a == 1 and b ==1:
                        effect = effect + 0.2*env[c1,c2]
                        env[c1,c2] = 0.8*env[c1,c2]
                    else:
                        effect = effect + 0.15*env[c1,c2]
                        env[c1,c2] = 0.85*env[c1,c2]
        self.env = env.copy()
        return env, effect