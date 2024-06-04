from typing import Tuple

import numpy as np

from ..dynamics import DubinsCar
from .robot import IRobot


class VEHICLE(IRobot):
    """Watering Cart."""
    def __init__(
        self,
        init_state: np.ndarray,
        water_volume: int,
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
        self.max_lin_vel = 1.0
        self.movements = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.goal_spray_flag = []
        self.spray_flag = False
        self.water_volume = water_volume
        self.water_volume_now = water_volume
        self.Water_replenishment_period = 2
        self.Water_replenishment_flag = False
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
        water_station: np.ndarray,
    ):
        """
        
        Parameters
        ----------
        goal_states: np.ndarray, shape=(num_states, dim_states)
        
        """
        if self.water_volume_now >= 1:
            self.goal_states = goal_states.copy()
            self.goal_spray_flag = goal_spray_flag.copy()
        else:
            self.goal_states = water_station.copy()
            goal_spray_flag = np.ones((1,1), dtype=bool)
            goal_spray_flag[0,0] = False
            self.goal_spray_flag = goal_spray_flag.copy()
            self.Water_replenishment_flag = True

    def update(self) -> None:
        """
        Update state, and goal states.

        """
        if self.has_goal:
            # Update state
            self.state = self.goal_states[0]
            self.spray_flag = self.goal_spray_flag[0]
            
            if self.spray_flag == True:
                self.water_volume_now = self.water_volume_now - 1
                
            if self.Water_replenishment_flag == True:
                self.Water_replenishment_period = self.Water_replenishment_period - 1
                
            if self.Water_replenishment_period == 0:
                self.water_volume_now = self.water_volume
                self.Water_replenishment_flag = False
                self.Water_replenishment_period = 2
        
            #update goal state
            self.goal_states = self.goal_states[1:]
            self.goal_spray_flag = self.goal_spray_flag[1:]
        
    def update_withoutreplenishment(self) -> None:
        """
        Update state, and goal states.

        """
        if self.has_goal:
            # Update state
            self.state = self.goal_states[0]
            self.spray_flag = self.goal_spray_flag[0]
        
            #update goal state
            self.goal_states = self.goal_states[1:]
            self.goal_spray_flag = self.goal_spray_flag[1:]
        

    def control(self) -> Tuple[float, np.ndarray]:
        """Compute control output, i.e. action.

        Returns
        -------
        dist: float
            Distance to the first goal state.
        action: np.ndarray
            Control output.

        """
        assert self.has_goal, "I need at least one goal state do control."

        x, y, o = self.state

        # Compute distance to the goal.
        goal_state = self.goal_states[0]
        goal_x, goal_y = goal_state[:2]
        x_diff = goal_x - x
        y_diff = goal_y - y
        dist = np.hypot(x_diff, y_diff)

        # Compute the goal position in the odometry frame.
        x_odom = np.cos(o) * x_diff + np.sin(o) * y_diff
        y_odom = -np.sin(o) * x_diff + np.cos(o) * y_diff

        linear_velocity = self.max_lin_vel * np.tanh(x_odom)
        # angular proportional parameter is set to 2.0
        angular_velocity = 2.0 * np.arctan2(y_odom, x_odom)

        action = np.array([linear_velocity, angular_velocity])
        return dist, action
