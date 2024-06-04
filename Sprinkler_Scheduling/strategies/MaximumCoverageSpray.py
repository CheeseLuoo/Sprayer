import copy
import random
import sys
import pickle as pkl
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List

# from sklearn.utils import shuffle
# from Common.utils import PrintExecutionTime

from ..gridcontext.GridMovingContext import GridMovingContext as GridMovingContext
# from ..gridcontext.GridMovingContext_MIdependSprayweight import GridMovingContext_MIDependSprayweight as GridMovingContext
from ..objectives.entropy import gaussian_entropy
from ..objectives.sprayeffect import spray_effect, calculate_effect
from ..models import IModel
from .strategy import IStrategy
from ..robots import IRobot

# return 0 if success, -1 if invalid
# a action is a tuple specifying time, move
#尝试移动位置，仅考虑位置移动
def try_move(context, agent, time, move):
  #获得智能体的动作列表
  MoveMatrix = context.GetMoveMatrices()
  #获得所选的智能体轨迹list=[time,action], time=[0:num],action=[x,y,volume,timestamp]
  agent_position_list = context.curr_trace_set[agent, :, :].copy()
  #获得所选智能体所选时刻的当前动作
  previous_policy = context.policy_matrix[agent].copy()
  #判断当前动作是否为补水状态，补水状态不可移动
  if previous_policy[time,2] == -1:
    return None, None
  #计算动作之间的差值(移动)
  move_diff = np.array(MoveMatrix[move][0:2]) - previous_policy[time, 0:2]
  #计算更新移动动作后的智能体轨迹，所选时刻后均会更新
  agent_position_list[(time + 1):, 0:2] += move_diff
  #检查轨迹是否有效
  if(context.CheckValid(agent_position_list)):
    new_policy = previous_policy.copy()
    new_policy[time, 0:2] = np.array(MoveMatrix[move][0:2])
    return agent_position_list[time + 1:, :], new_policy
  else:
    return None, None
  
#尝试改变洒水方式0，将原状态改为不洒水，在补水前随机插入洒水动作
#对于补水时刻的移动动作调整，有两种方式，1.在新插入的洒水动作处的添加一个停留移动动作，其余动作依次顺延
#2.不插入动作，将后续的补水处的停留均后移，这里选择第二种
def try_spray0(rng, context, agent, selecttime):
  previous_policy = context.policy_matrix[agent].copy()
  num = 0
  time = 0
  for i in range(context.GetMaxTime()):
    if previous_policy[i,2] == 1:
      num = num + 1
    if num == selecttime + 1:
      time = i
      break
  if previous_policy[time,2] != 1:
    return None
  else:
    # 当前状态洒水时，搜索之后的第一个补水动作时段,然后在之前随机插入一个洒水动作
    # 确定第一个补水时间
    replenish_time = 0
    for i in range(context.GetMaxTime()-time-1):
      action = previous_policy[time+i+1]
      if action[2] == -1:
        replenish_time = time + i + 1
        break
    if replenish_time == 0:
      return None
    rand_replenish_time = rng.randint(time + 1, replenish_time + 1)
    
    # 计算变更后的policy
    new_policy = previous_policy.copy()
    # 先调整补水动作
    rep = 0
    for i in range(context.GetMaxTime()- time - 1):
      if new_policy[time + i + 1, 2] == -1:
        rep = rep + 1
        continue
      if rep >= 1:
        # 交换
        new_policy[time + i + 1 - rep, 0] = new_policy[time + i + 1, 0]
        new_policy[time + i + 1 - rep, 1] = new_policy[time + i + 1, 1]
        new_policy[time + i + 1, 0] = 0
        new_policy[time + i + 1, 1] = 0
        rep = 0
    
    # 再调整洒水动作
    new_policy[time,2] = 0
    # 将插入的补水时间的动作后的动作依次顺延, 注意从前向后顺延，将会复制前面的策略
    for i in range(context.GetMaxTime() - rand_replenish_time - 1):
      m = context.GetMaxTime() - rand_replenish_time - 2 - i
      new_policy[rand_replenish_time + m + 1,2] = new_policy[rand_replenish_time + m,2]
      # new_policy[rand_replenish_time + m + 1,0] = new_policy[rand_replenish_time + m,0]
      # new_policy[rand_replenish_time + m + 1,1] = new_policy[rand_replenish_time + m,1]
    new_policy[rand_replenish_time,2] = 1
    # new_policy[rand_replenish_time,0] = 0
    # new_policy[rand_replenish_time,1] = 0
    return new_policy

#尝试改变洒水方式1，随机选择一个动作，当该动作为不洒水动作时，将该不洒水动作移除，后续动作均前移，在最后根据车辆状态插入动作
# 对于补水时的停留动作，有两种修复方式，第一种，将不洒水时刻的移动动作同步移除，在最后加入停留动作；第二种，将不洒水后的补水停留均前移。
# 第一种的实现较为简单，这里选择第二种
def try_spray1(rng, context, agent, selecttime):
  agent_position_list = context.curr_trace_set[agent, :, :].copy()
  previous_policy = context.policy_matrix[agent].copy()
  num = 0
  time = 0
  for i in range(context.GetMaxTime()):
    if previous_policy[i,2] == 0:
      num = num + 1
    if num == selecttime + 1:
      time = i
      break
  if previous_policy[time,2] == 0:
    # 计算变更后的policy
    new_policy = previous_policy.copy()
    # 将选择的不洒水动作后的补水动作处的移动动作前移(交换)
    for i in range(context.GetMaxTime()- time - 1):
      if new_policy[time + i + 1, 2] == -1:
        # 与前面的移动动作交换(把前面的动作拿过来，把前面变成不移动)
        new_policy[time + i + 1, 0] = new_policy[time + i, 0]
        new_policy[time + i + 1, 1] = new_policy[time + i, 1]
        new_policy[time + i, 0] = 0
        new_policy[time + i, 1] = 0
    
    # 将选中时间后的洒水动作依次顺延
    for i in range(context.GetMaxTime()- time - 1):
      new_policy[time + i, 2] = new_policy[time + i + 1, 2]
      
    # 在最后插入合适的动作
    if agent_position_list[context.GetMaxTime(), 2] >= 1 and previous_policy[context.GetMaxTime() - 1, 2] != -1:
      new_policy[context.GetMaxTime() - 1, 2] = 1
    elif agent_position_list[context.GetMaxTime(), 2] < 1:
      new_policy[context.GetMaxTime() - 1, 2] = -1
      new_policy[context.GetMaxTime() - 1, 0] = 0
      new_policy[context.GetMaxTime() - 1, 1] = 0
    elif agent_position_list[context.GetMaxTime(), 2] >= context.Setting.water_volume:
      new_policy[context.GetMaxTime() - 1, 2] = 1
    elif agent_position_list[context.GetMaxTime(), 2] < context.Setting.water_volume and previous_policy[context.GetMaxTime() - 1, 2] == -1:
      new_policy[context.GetMaxTime() - 1, 2] = -1
      new_policy[context.GetMaxTime() - 1, 0] = 0
      new_policy[context.GetMaxTime() - 1, 1] = 0
    return new_policy
  else:
    return None
  
#尝试改变洒水方式2，在一个洒水周期内移动非洒水动作的位置
def try_spray2(rng, context, agent, selecttime):
  previous_policy = context.policy_matrix[agent].copy()
  num = 0
  time = 0
  for i in range(context.GetMaxTime()):
    if previous_policy[i,2] == 0:
      num = num + 1
    if num == selecttime + 1:
      time = i
      break
    
  if previous_policy[time,2] == 0:
    # 寻找前后两个补水时刻
    replenish_time_1 = 0#前一个补水时刻
    for i in range(time):
      action = previous_policy[time-i-1]
      if action[2] == -1:
        replenish_time_1 = time-i-1
        break
      if i == time - 1:
        replenish_time_1 = -1
    
    replenish_time_2 = 0
    for i in range(context.GetMaxTime()-time-1):
      action = previous_policy[time+i+1]
      if action[2] == -1:
        replenish_time_2 = time + i + 1
        break
      if i == context.GetMaxTime()-time-2:
        replenish_time_2 = context.GetMaxTime()
    if replenish_time_2 - replenish_time_1 <= 1:
      return None
    
    # 统计该洒水阶段内的所有洒水次数
    num = 0
    for i in range(replenish_time_2 - replenish_time_1 - 1):
      if previous_policy[replenish_time_1 + 1 + i,2] == 1:
        num = num + 1
    
    if num == 0:
      return None
    
    # 寻找准备交换的洒水时段
    rand_exchange_time = rng.randint(0, num)
    exchange_time = 0
    for i in range(context.GetMaxTime()):
      if previous_policy[replenish_time_1 + 1 + i,2] == 1:
        num = num - 1
      if num == rand_exchange_time:
        exchange_time = replenish_time_1 + 1 + i
        break
    if previous_policy[exchange_time,2] != 1:
      return None
    # 计算变更后的policy
    new_policy = previous_policy.copy()
    new_policy[time,2] = 1
    new_policy[exchange_time,2] = 0
    return new_policy
  else:
    return None
    
def do_move(context, agent, time, New_policy, agent_position_list) -> GridMovingContext:
  # MoveMatrix = context.GetMoveMatrices()
  context.policy_matrix[agent] = New_policy
  context.curr_trace_set[agent, time + 1:, :] = agent_position_list
  
def do_spray(context, agent, New_policy) -> GridMovingContext:
  context.policy_matrix[agent] = New_policy
  for i in range(context.GetAgentNumber()):
    for j in range(context.GetMaxTime() + 1):
      if(j == 0):
        continue
      else:# 注意，洒水时，由于调整了移动，因此位置循序也要变化，这种变化可以在设计变化逻辑时考虑，也可以在实施变化时统一考虑
        #这里选择在这里统一考虑
        # 洒水
        # print(context.policy_matrix)
        context.curr_trace_set[i, j, 0:2] = context.curr_trace_set[i, j - 1, 0:2] + np.array(context.policy_matrix[i, j - 1, 0:2])
        if context.policy_matrix[i, j - 1, 2] == 1:
          context.curr_trace_set[i, j, 2] = context.curr_trace_set[i, j - 1, 2] - context.policy_matrix[i, j - 1, 2]
        # 补水
        elif context.policy_matrix[i, j - 1, 2] == -1:
          context.curr_trace_set[i, j, 2] = context.curr_trace_set[i, j - 1, 2] - context.policy_matrix[i, j - 1, 2] * context.Setting.replenish_speed
          if context.curr_trace_set[i, j, 2] >= context.Setting.water_volume:
            context.curr_trace_set[i, j, 2] = context.Setting.water_volume
        # 其他
        else:
          context.curr_trace_set[i, j, 2] = context.curr_trace_set[i, j - 1, 2]

# def SimulatedAnnealing(origin_mc_context: GridMovingContext, *, n_playout=10000, initial_temp=1, k=0.95, bound=100, min_temp= 0.001, mini_step=1):
def SimulatedAnnealing(rng, origin_mc_context: GridMovingContext, *,enough_info = None, n_playout=10000, initial_temp=1, k=0.95, bound=100, min_temp= 0.001, 
                       mini_step=1, object = 1, object_mi = 50):
  # 注意，仅考虑顺序洒水，以信息为目标进行路径规划，不考虑选择性洒水，仅进行位置移动
  sq_list = []
  curr_turns = 0
  curr_k = 1
  # try:
  Temp = initial_temp
  curr_context = copy.deepcopy(origin_mc_context)
  # seed = curr_context.Setting.seed
  # random.seed(seed)
  while(curr_turns < bound):
    iters = 0
    curr_turns += 1
    if(curr_k >= min_temp):
      curr_k = k * curr_k
    # print(curr_turns)
    while(iters < n_playout):
      iters += 1
      # rand_category = rng.randint(0, 1)
      rand_category = 0
      rand_spray_category = rng.randint(0, 3)
      rand_agent = rng.randint(0, curr_context.GetAgentNumber())
      rand_time = rng.randint(0, curr_context.GetMaxTime())
      SprayTime = curr_context.GetSprayTime(rand_agent)
      if SprayTime == 0:
        rand_time1 = 0
      else:
        rand_time1 = rng.randint(0, SprayTime)
      DontSprayTime = curr_context.GetDontSprayTime(rand_agent)
      if DontSprayTime == 0:
        rand_time2 = 0
      else:
        rand_time2 = rng.randint(0, DontSprayTime)
      rand_action = rng.randint(0, curr_context.GetPossibleActions())
      agent_position_list = None
      New_policy = None
      if rand_category == 0:
        # 调整位置
        agent_position_list, New_policy = try_move(curr_context, rand_agent, rand_time, rand_action)
      else:
        # 调整洒水动作
        if rand_spray_category == 0:
          New_policy = try_spray0(rng, curr_context, rand_agent, rand_time1)
        elif rand_spray_category == 1:
          New_policy = try_spray1(rng, curr_context, rand_agent, rand_time2) 
        elif rand_spray_category == 2:
          New_policy = try_spray2(rng, curr_context, rand_agent, rand_time2)
      
      # 分类执行
      # 根据随机选择的动作操作智能体轨迹
      new_mc_context = copy.deepcopy(curr_context)
      # print(rand_category,rand_spray_category)
      if rand_category == 0:
        # 调整位置
        if(agent_position_list is None):
          continue
        do_move(new_mc_context, rand_agent, rand_time, New_policy, agent_position_list)
      elif rand_category == 1:
        # 调整洒水
        if rand_spray_category == 0:
          if(New_policy is None):
            continue
          do_spray(new_mc_context, rand_agent, New_policy)  
        elif rand_spray_category == 1:
          if(New_policy is None):
            continue
          do_spray(new_mc_context, rand_agent, New_policy)  
        elif rand_spray_category == 2:
          if(New_policy is None):
            continue
          do_spray(new_mc_context, rand_agent, New_policy)  

      # 仅使用信息目标作为接收标准
      if object == 1:
        MI_before = curr_context.CalculateMISQ()
        MI_after = new_mc_context.CalculateMISQ()
        delta_MI = MI_after - MI_before
        if(delta_MI >= 0):
          curr_context = new_mc_context
        else:
          # accept by chance
          accept_prob = np.exp(delta_MI / (curr_k * Temp))
          if(rng.random() < accept_prob):
            curr_context = new_mc_context
      # 仅使用洒水目标作为接收标准
      elif object == 2:
        sprayeffect_before, _ = curr_context.calculate_Sprayscores_foreveryvehicle()
        sprayeffect_after, _ = new_mc_context.calculate_Sprayscores_foreveryvehicle()
        delta_sprayeffect = sprayeffect_after - sprayeffect_before
        if(delta_sprayeffect >= 0):
          curr_context = new_mc_context
        else:
          # accept by chance
          accept_prob = np.exp(delta_sprayeffect / (curr_k * Temp))
          if(rng.random() < accept_prob):
            curr_context = new_mc_context
      
      # 综合的接收情况，分别通过，洒水限制优先考虑
      elif object == 3:
        # MI_before = curr_context.CalculateMISQ()
        # MI_after = new_mc_context.CalculateMISQ()
        # spray_time_before = curr_context.calculate_wolume_scores()
        # spray_time_after = new_mc_context.calculate_wolume_scores()
        
        # delta_MI = MI_after - MI_before
        # delta_spraytime = spray_time_after - spray_time_before
        MI_after = 100
        delta_MI = 0
        delta_spraytime = 0
        if enough_info[rand_agent] == False:
          print(1)
          if delta_spraytime <= 0:
            if rand_category == 0:
              delta_MI = np.max((8-rand_time,1))*delta_MI
            if(delta_MI >= 0):
              curr_context = new_mc_context
            else:
              # accept by chance
              accept_prob = np.exp(delta_MI / (curr_k * Temp[0]))
              if(rng.random() < accept_prob):
                curr_context = new_mc_context
        else:
          if MI_after < object_mi[rand_agent]:
            print(2)
            if delta_MI >= 0:
              curr_context = new_mc_context
            elif delta_MI < 0:
              accept_prob = np.exp(delta_MI / (curr_k * Temp[0]))
              if(rng.random() < accept_prob):
                curr_context = new_mc_context
            
          elif MI_after >= object_mi[rand_agent]:
            sprayeffect_before = curr_context.CalculateSpraySQ()
            sprayeffect_after = new_mc_context.CalculateSpraySQ()
            delta_sprayeffect = sprayeffect_after - sprayeffect_before
            if delta_sprayeffect >= 0:
              curr_context = new_mc_context
            elif delta_sprayeffect < 0:
              print(delta_sprayeffect)
              accept_prob = np.exp(delta_sprayeffect / (curr_k * Temp[1]))
              if(rng.random() < accept_prob):
                curr_context = new_mc_context
      
    sprayeffect_after = curr_context.CalculateSpraySQ()
    sprayeffect_after5step, _ = curr_context.calculate_Sprayscores_foreveryvehicle()
    MI_after = curr_context.CalculateMISQ()
    sq_list.append((sprayeffect_after, sprayeffect_after5step, MI_after))
  return curr_context, sq_list

# @PrintExecutionTime
def SimulatedAnnealingFixed(rng, origin_context: GridMovingContext, bound, alpha):
  # 洒水车顺序规划算法，假设环境已知，以信息目标为通过标准计算
  # 计算当前的分数并储存
  sprayeffect_before = origin_context.CalculateSpraySQ()
  sprayeffect_before5step, _ = origin_context.calculate_Sprayscores_foreveryvehicle()
  mi_low = origin_context.CalculateMISQ()
  sq_list_total = []
  sq_list_total.append((sprayeffect_before, sprayeffect_before5step, mi_low))
  # sq_list_total.append((sprayeffect_before, 0, 0))

  # 以信息目标为单一目标进行综合规划
  single_playout = origin_context.GetAgentNumber() * origin_context.GetMaxTime()
  Info_Temp = np.max((30 - origin_context.Setting.current_step * 10,8))
  Temp = 30
  k = math.pow(0.00002, 1 / bound)
  context, sq_list = SimulatedAnnealing(rng,origin_context, n_playout = single_playout, initial_temp = Temp, k = k, bound = bound, object = 1)
  # print(context.policy_matrix)
  # print(context.curr_trace_set)
  return context, sq_list_total + sq_list

#定义SA算法包装
class SAMaximumCoverageSpray(IStrategy):
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
          顺序执行，根据车辆数相互配合，避免重复区域，实现上，以信息最大为目标，仅调整位置,计算时固定规划长度为8。
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
        # 计算当前需要规划的步数
        sche_step = 0
        if Setting.current_step > Setting.max_num_samples - Setting.sche_step:
          if Setting.max_num_samples - Setting.current_step > 7:
            sche_step = Setting.max_num_samples - Setting.current_step
          else:
            sche_step = 8
        else:
          sche_step = Setting.sche_step
          
        # 计算用于规划的目标集合 阶梯式的非均匀
        allpoint_list = []
        a = ((np.ceil((self.task_extent[1]-self.task_extent[0])/2)*2)-(self.task_extent[1]-self.task_extent[0]-1))/2
        b = ((np.ceil((self.task_extent[1]-self.task_extent[0])/3)*3)-(self.task_extent[1]-self.task_extent[0]-1))/2
        
        for num in range(0,Setting.sche_step,2):
          for i in np.arange (self.task_extent[0]-a,self.task_extent[1]+a,2):
            for j in np.arange (self.task_extent[2]-a,self.task_extent[3]+a,2):
              allpoint_list.append([i, j, model.time_stamp + num * Setting.time_co])
        allpoint = np.array(allpoint_list)
        
        if self.moving_context is None:
          agent_init_position = []
          for id, vehicle in self.vehicle_team.items():
            agent_init_position.append(vehicle.state[0:2])
          agent_init_position = np.array(agent_init_position)
          self.moving_context = GridMovingContext(agent_init_position, model, pred, allpoint, Setting)
          self.alpha = Setting.alpha
        else:
          self.moving_context.adaptive_update(model, pred, allpoint, Setting)
        self.moving_context, sq_list_total = SimulatedAnnealingFixed(self.rng, self.moving_context, Setting.bound, self.alpha)
        
        #context中包含最后的结果
        policy_now = self.moving_context.policy_matrix.copy()
        agent_position_list = self.moving_context.curr_trace_set.copy()
                
        result = dict()

        for id, vehicle in self.vehicle_team.items():
          goal_states = np.zeros((sche_step,2))
          spray_states = np.ones((sche_step,1))

          # Append waypoint
          for index in range(sche_step):
            goal_states[index,0] = agent_position_list[id-1,index+1,0]
            goal_states[index,1] = agent_position_list[id-1,index+1,1]
            spray_states[index,0] = policy_now[id-1,index,2]
            
          result[id] = (goal_states,spray_states)
                
        return result
    