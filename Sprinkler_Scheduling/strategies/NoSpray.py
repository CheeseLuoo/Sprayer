import copy
import random
import sys
import pickle as pkl
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List



from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot



#定义SA算法包装
class NoSpray(IStrategy):
    """Sequential planning based on real world experience on lattice map."""

    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        vehicle_team: dict,
    ) -> None:
        """
        Parameters
        ----------
        task_extent: List[float], [xmin, xmax, ymin, ymax]
            Bounding box of the sampling task workspace.shou
        rng: np.random.RandomState
            Random number generator if `get` has random operations.
        vehicle_team: dict
            team of vehicle.

        """
        super().__init__(task_extent, rng)
        self.vehicle_team = vehicle_team
        self.moving_context = None

        
    def get(self, model: IModel, Setting, pred) -> np.ndarray:
        """Get goal states.
          不移动,不洒水,规划长度规定为8
        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        Setting: Congif Class

        Returns
        -------
        result: dict, id:(goal_states,spray_flag)
            Sampling goal states and spray_flag

        """
        print('current_turn')
        print((Setting.current_step + Setting.adaptive_step)/Setting.adaptive_step)
        
        Setting.sche_step = 10
        sche_step = Setting.sche_step
        result = dict()

        for id, vehicle in self.vehicle_team.items():
          goal_states = np.zeros((sche_step,2))
          spray_states = np.zeros((sche_step,1))

          # Append waypoint
          for index in range(sche_step):
            goal_states[index,0] = vehicle.state[0]
            goal_states[index,1] = vehicle.state[1]
            
          result[id] = (goal_states,spray_states)
                
        return result
    