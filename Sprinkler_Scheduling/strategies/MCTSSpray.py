import copy
import random
import sys
import pickle as pkl
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List
import matplotlib.pyplot as plt
import itertools
# from sklearn.utils import shuffle
# from Common.utils import PrintExecutionTime

from ..gridcontext.MCTSContext import MCTSContext
from ..gridcontext.MCTS_concurrent import MCTSConcurrentPlayer
# from ..gridcontext.GridMovingContext_MIdependSprayweight import GridMovingContext_MIDependSprayweight as GridMovingContext
from ..objectives.entropy import gaussian_entropy
from ..objectives.sprayeffect import spray_effect, calculate_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

# 定义了一个均匀概率的函数
def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0

#定义SA算法包装
class MCTSSpray(IStrategy):
    """MCTS case on sprinkler scheduling."""

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
        """Get goal states for sampling.

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
        test_map_size = (Setting.task_extent[1], Setting.task_extent[3])
        agent_init_position = []
        for id, vehicle in self.vehicle_team.items():
          agent_init_position.append(vehicle.state[0:2])
        agent_init_position = np.array(agent_init_position)
        test_positions = agent_init_position
        test_number = agent_init_position.shape[0]
        print(test_positions)
        print(test_number)

        # 初始化context
        initial_state = MCTSContext(test_map_size, Setting.water_volume, 
                                    agent_init_position=test_positions,
                                    agent_number=test_number,
                                    pollution_distribute = pred)

        player = MCTSConcurrentPlayer(policy_value_fn, n_playout=Setting.bound1)

        # 搜索
        states = []
        while True:
            move = player.get_action(initial_state)
            states.append(initial_state.agent_curr_position.copy())
            initial_state.do_move(move)
            game_end, sq = initial_state.game_end()
            # print(len(initial_state.record))
            if(game_end):
                # print(initial_state.calculate_trace())
                break
        print(initial_state.record,sq)
        # sys,exit()
        result = dict()
        schestep = np.ceil(Setting.water_volume + Setting.water_volume/Setting.replenish_speed).astype(int)
        agent_position = agent_init_position.copy()
        MoveMatrix2D = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
        for id, vehicle in self.vehicle_team.items():
          goal_states = np.zeros((schestep, 2))
          spray_states = np.ones((schestep, 1))
          for t in range(Setting.water_volume):
            if(t * test_number + id - 1 >= len(initial_state.record)):
                break
            direction = MoveMatrix2D[initial_state.record[t * test_number + id - 1]]
            agent_position[id - 1] = agent_position[id - 1] + np.array(direction)
            goal_states[t,0] = agent_position[id - 1][0]
            goal_states[t,1] = agent_position[id - 1][1]
            spray_states[t,0] = 1

          for index in range(schestep):
            if index >= Setting.water_volume:
              goal_states[index,0] = agent_position[id - 1][0]
              goal_states[index,1] = agent_position[id - 1][1]
              spray_states[index,0] = -1
              
          result[id] = (goal_states,spray_states)
                
        return result
    