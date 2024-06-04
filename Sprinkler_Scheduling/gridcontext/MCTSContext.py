import itertools
import numpy as np
import sys
# sys.path.append("..") 
# sys.path.append("quest_scheduler")
MoveMatrix2D = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
from ..objectives.sprayeffect import spray_effect, calculate_effect

def even_dist(map_shape):
  dst = np.ones(map_shape)
  return dst / np.sum(dst)

class MCTSContext():
    def __init__(self, map_shape, time, **kwargs) -> None:
        self.map_shape = map_shape
        self.time = time
        self.agent_init_position = kwargs['agent_init_position']
        self.agent_number = kwargs['agent_number']
        self.current_player = 0
        self.agent_curr_position = [] # 依次存放了agent的当前位置
        self.availables = [] # 记录当前agent的可行动位置
        self.record = [] # 依次记录了每个agent执行的动作
        self.side = [] # 依次记录了执行动作的agent顺序
        self.init_agent_trace()
        self.target_dist = even_dist(map_shape)
        self.pollution_distribute = kwargs['pollution_distribute']
        # self.pollution_distribute = None

    def init_agent_trace(self):
        for i_player in range(self.agent_number):
            self.agent_curr_position.append(self.agent_init_position[i_player])
        self.agent_curr_position = np.array(self.agent_curr_position)
        self.availables = self.calculate_availables()

    # action是一个数字，代表movematrix中的动作序号
    def do_move(self, action):
        self.record.append(action)
        self.agent_curr_position[self.current_player] = \
        self.agent_curr_position[self.current_player] + np.array(MoveMatrix2D[action])
        self.current_player += 1
        if(self.current_player >= self.agent_number):
            self.current_player = 0
        self.availables = self.calculate_availables()

    # 返回一个列表，move一次，side中加入一次player，计算当前player下一次移动的可选位置,返回的位置用数字代替
    def calculate_availables(self):
        # print("context")
        # print(self.agent_curr_position)
        # print(self.current_player)
        curr_position = self.agent_curr_position[self.current_player]
        self.side.append(self.current_player)
        available_positions = []
        for i_direction in range(len(MoveMatrix2D)):
            direction =  MoveMatrix2D[i_direction]
            new_posision = np.array(curr_position) + np.array(direction)
            if(new_posision[0] < 0 or new_posision[0] >= self.map_shape[0]):
                continue
            if(new_posision[1] < 0 or new_posision[1] >= self.map_shape[1]):
                continue
            available_positions.append(i_direction)
        return available_positions

    def game_end(self):
        if(len(self.record) >= self.time * self.agent_number):
            return True, self.calculate_sq()
        else:
            return False, -1
      
    # 从record中读数据，将轨迹映射到矩阵中，用于计算收益
    def calculate_matrix(self):
        ground_matrix = np.zeros(self.map_shape)
        agent_position = self.agent_init_position.copy().astype(int)
        for i_player in range(self.agent_number):
            ground_matrix[agent_position[i_player][0], agent_position[i_player][1]] += 1

        for t in range(self.time):
            for i_player in range(self.agent_number):
                if(t * self.agent_number + i_player >= len(self.record)):
                    break
                data_value = 1
                direction = MoveMatrix2D[self.record[t * self.agent_number + i_player]]
                new_position = np.array(agent_position[i_player]) + np.array(direction)
                ground_matrix[new_position[0], new_position[1]] += data_value
                agent_position[i_player] = new_position
        return ground_matrix

    def calculate_sq(self):
        return self.calculate_Spray_scores()
    
    def calculate_kldiv(self):
        ground_matrix = self.calculate_matrix()
        x = ground_matrix / np.sum(ground_matrix)
        non_zero_mask = x!=0
        kldiv =  np.sum( x[non_zero_mask] * np.log(x / self.target_dist)[non_zero_mask])
        # return kldiv
        return 0
    
    def calculate_Spray_scores(self):
        spray_effect = 0
        pollution_distribute = self.pollution_distribute.copy()
        spray_done = np.ones((self.agent_number, pollution_distribute.shape[0], pollution_distribute.shape[1]))
        agent_position = self.agent_init_position.copy()
        for t in range(self.time):
            for i_player in range(self.agent_number):
                if(t * self.agent_number + i_player >= len(self.record)):
                    break
                direction = MoveMatrix2D[self.record[t * self.agent_number + i_player]]
                new_position = np.array(agent_position[i_player]) + np.array(direction)
                r0 = new_position[0]
                c0 = new_position[1]
                for a in range(5):
                    for b in range(5):
                        r = int(r0 - 2 + a) 
                        c = int(c0 - 2 + b)
                        effect_rate = 1
                        if r >= 0 and r < self.map_shape[0] and c >= 0 and c < self.map_shape[1]:
                            if a == 2 and b == 2:
                                spray_effect = spray_effect + spray_done[i_player,r,c]*0.15*(12-t)*calculate_effect(pollution_distribute[r,c])
                                pollution_distribute[r,c] = pollution_distribute[r,c] - effect_rate*calculate_effect(pollution_distribute[r,c])
                            elif (a - 2)**2 + (b - 2)**2 <= 2:
                                spray_effect = spray_effect + spray_done[i_player,r,c]*0.7*0.15*(12-t)*calculate_effect(pollution_distribute[r,c])
                                pollution_distribute[r,c] = pollution_distribute[r,c] - 0.7*effect_rate*calculate_effect(pollution_distribute[r,c])
                            else:
                                spray_effect = spray_effect + spray_done[i_player,r,c]*0.5*0.15*(12-t)*calculate_effect(pollution_distribute[r,c])
                                pollution_distribute[r,c] = pollution_distribute[r,c] - 0.5*effect_rate*calculate_effect(pollution_distribute[r,c])
                agent_position[i_player] = new_position     
        return spray_effect