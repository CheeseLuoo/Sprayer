from pathlib import Path
from typing import List, Tuple

import numpy as np

class Evaluator:
    """Evaluate the performance of a model."""

    def __init__(self) -> None:
        pass

    def eval_results(self, env, task_extent, team):
        coverage = self.compute_coverage(env, task_extent, team)
        mean_airpollution, max_airpollution = self.compute_airpollution(env)
        
        return coverage, mean_airpollution, max_airpollution
    
    def compute_coverage(self, env, task_extent, team):
        # 计算agent的位置对env高浓度区域的覆盖率
        binary_env = env > 60
        count_total = np.sum(binary_env)
        if count_total > 0:
            # 计算覆盖坐标，取周围一圈
            surrounding_coords = []
            for _, agent in team.items():
                r = agent.state[0]
                c = agent.state[1]
                # 遍历当前车辆周围一圈的所有坐标
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        # 计算当前坐标
                        coord = [int(r + i), int(c + j)]
                        # 判断当前坐标是否已经存在
                        if tuple(coord) in [tuple(x) for x in surrounding_coords]:
                            continue
                        # 排除超出范围的坐标
                        if task_extent[0] <= coord[0] < task_extent[1] and task_extent[2] <= coord[1] < task_extent[3]:
                            # 添加到坐标集合中
                            surrounding_coords.append(coord)
            # 计算覆盖坐标中有效位置的数量    
            count_coverage = np.sum(binary_env[[coord[0] for coord in surrounding_coords],[coord[1] for coord in surrounding_coords]])
            coverage = count_coverage / count_total
        else:
            coverage = 0
        return coverage

    def compute_airpollution(self, env):
        mean_airpollution = np.mean(env)
        max_airpollution = np.max(env)
        
        return mean_airpollution, max_airpollution