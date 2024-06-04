from typing import Tuple

import numpy as np

from ..dynamics import DubinsCar
from .robot import IRobot


class SPRINKLER_REPLENISHANYWHERE(IRobot):
    """Watering Cart."""
    def __init__(
        self,
        init_state: np.ndarray,
        Setting,
    ) -> None:
        """

        Parameters
        ----------
        init_state: np.ndarray, shape=(dim_states, ), dtype=np.float64
            Initial robot state.

        """
        self._check_inputs(
            init_state,
        )
        dynamics = DubinsCar(10)
        super().__init__(init_state, dynamics, 0.1, 5)
        # self.max_lin_vel = 1.0
        self.movements = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.goal_spray_flag = []
        self.spray_flag = False
        
        self.water_volume = Setting.water_volume
        self.water_volume_now = Setting.water_volume
        self.replenish_speed = Setting.replenish_speed

        self.traj = np.zeros((1,2))
        self.traj[0,0] = init_state[0]
        self.traj[0,1] = init_state[1]

    @staticmethod
    def _check_inputs(
        init_state: np.ndarray,
    ):
        """

        Parameters
        ----------
        init_state: np.ndarray, shape=(dim_states, ), dtype=np.float64
            Initial robot state.

        """
        if init_state.ndim != 1 or init_state.dtype != np.float64:
            raise ValueError("init_state: np.ndarray, " +
                             "shape=(dim_states, ), dtype=np.float64")
            
            
    def set_goals(self,
        goal_states: np.ndarray,
        goal_spray_flag: np.ndarray,
    ):
        """
        
        Parameters
        ----------
        goal_states: np.ndarray, shape=(num_states, dim_states)
        
        """
        self.goal_states = goal_states.copy()
        self.goal_spray_flag = goal_spray_flag.copy()


    def update(self) -> None:
        """
        Update state, and goal states.

        """
        if self.has_goal:
            # Update state
            self.state = self.goal_states[0]
            self.spray_flag = self.goal_spray_flag[0]
            
            if self.spray_flag == 1:
                self.water_volume_now = self.water_volume_now - 1
            elif self.spray_flag == -1:
                self.water_volume_now = self.water_volume_now + self.replenish_speed
            elif self.spray_flag == 0:
                self.water_volume_now = self.water_volume_now
                
            #update goal state
            self.goal_states = self.goal_states[1:]
            self.goal_spray_flag = self.goal_spray_flag[1:]
        else:
            print('without_goals')
            
    def control(self):
        return None
        
