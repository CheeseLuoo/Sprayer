import copy
import random
import sys
import pickle as pkl
from time import time

import itertools
import numpy as np
import math
# from ..Common.QFunctions import time_decay_aggregation, calculate_kldiv
# from ..Common.common import even_dist, gaussian_dist, dg_corner_dist, noise_dist
from ..models import IModel
from ..objectives.entropy import gaussian_entropy, gaussian_entropy_multivariate
from ..objectives.sprayeffect import spray_effect, calculate_effect
MoveMatrix2D = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
MoveMatrix2DWithSprayControl = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))#暂时假设每一次均只能选择一步，并选择是否洒水，1：洒水，-1：补水
SprayControl = list(itertools.product([-1, 0, 1]))

class GridMovingContext():
  def __init__(self, agent_init_position, model:IModel, pollution_distribute, allpoint, Setting, move_matrix = MoveMatrix2DWithSprayControl, alpha = 0.5) -> None:
    """
    Accepted keywords:
    target_dist: A mapshape sized map, indicating Object data value distributions.
    NOTE: the length of trace set is 1 more than time
    """
    #参数
    self.Setting = Setting
    if Setting.current_step > Setting.max_num_samples - Setting.sche_step:
      if Setting.max_num_samples - Setting.current_step > 7:
        self.Setting.sche_step = Setting.max_num_samples - Setting.current_step
      else:
        self.Setting.sche_step = 8
    else:
      self.Setting.sche_step = Setting.sche_step
    #作业区域
    self.map_shape = (Setting.task_extent[1], Setting.task_extent[3])
    #规划时长
    self.time = self.Setting.sche_step
    #初始智能体位置
    self.agent_init_position = agent_init_position
    #智能体的移动模型
    self.move_matrix = move_matrix
    #预测模型及所需参数
    self.model = model
    self.alpha = alpha
    self.pollution_distribute = pollution_distribute
    # 计算峰值污染区域（固定）
    indices = np.argpartition(pollution_distribute, -4, axis=None)[-4:]
    row_indices, col_indices = np.unravel_index(indices, pollution_distribute.shape)
    # 计算最大元素以及周围一圈元素的浓度均值
    self.sourcelocations = set()
    for i in range(len(row_indices)):
      max_row = row_indices[i]
      max_col = col_indices[i]
      # 计算周围一圈元素的位置
      for row_offset in range(-2, 3):
        for col_offset in range(-2, 3):
          surrounding_row = max_row + row_offset
          surrounding_col = max_col + col_offset
          if 0 <= surrounding_row < pollution_distribute.shape[0] and 0 <= surrounding_col < pollution_distribute.shape[1]:
            self.sourcelocations.add((surrounding_row, surrounding_col))
              
    self.allpoint = allpoint
    self.agent_number = self.agent_init_position.shape[0]
    print("agent_number")
    print(self.agent_number)
    self.time_co = Setting.time_co
    self.last_info = 0.5
    
    #智能体策略矩阵
    # 定义初始动作，原地连续洒水，策略矩阵不再存放动作标号，而直接存放动作
    self.policy_matrix = self.init_policy_matrix()
    #智能体轨迹，存放了包括初始轨迹在内，轨迹是一系列二维坐标
    self.curr_trace_set = self.calculate_trace_set()
    #当前轨迹所覆盖矩阵
    self.curr_matrixA, self.curr_matrixB, self.curr_matrixC = self.calculate_matrix()
    #可选动作长度
    self.possible_actions = len(self.move_matrix)
    
    #计算时刻已观测点对当前层各点的信息量
    # self.MIforeverypoint
    allpoint_list = []
    for i in range (self.Setting.task_extent[0],self.Setting.task_extent[1]):
        for j in range (self.Setting.task_extent[2],self.Setting.task_extent[3]):
            allpoint_list.append([i, j, self.model.time_stamp])
    point = np.array(allpoint_list)
    prior_diag_std, poste_diag_std, _, _ = self.model.prior_poste(point)
    hprior = gaussian_entropy(prior_diag_std.ravel())
    hposterior = gaussian_entropy(poste_diag_std.ravel())
    mi_all = hprior - hposterior
    if np.any(mi_all < 0.0):
        print(mi_all.ravel())
        raise ValueError("Predictive MI < 0.0!")
    normed_mi = (mi_all - mi_all.min()) / mi_all.ptp()
    MI_information = np.zeros((self.Setting.task_extent[1]-self.Setting.task_extent[0],self.Setting.task_extent[3]-self.Setting.task_extent[2]))
    for i in range (self.Setting.task_extent[0],self.Setting.task_extent[1]):
        for j in range (self.Setting.task_extent[2],self.Setting.task_extent[3]):
            MI_information[i,j] = normed_mi[i*(self.Setting.task_extent[3]-self.Setting.task_extent[2])+j]
    self.MIforeverypoint = MI_information
  
  def init_policy_matrix(self):
    #初始策略矩阵，根据车辆的水量和补水速度计算
    replenish_speed = self.Setting.replenish_speed
    water_volume = self.Setting.water_volume
    replenish_time = math.ceil(water_volume/replenish_speed)
    round = math.ceil(self.time/(replenish_time + water_volume))
    line = []
    for i in range(round):
      for j in range(water_volume):
        line.append([0, 0, 1])
        
      for j in range(replenish_time):
        line.append([0, 0, -1])
    line = line[0:self.time]
    line = np.array(line)
    mid_line = np.ones((self.agent_number,line.shape[0],line.shape[1]))
    for j in range(self.agent_number):
      mid_line[j,:,:] = line
    return mid_line

  def CalculateSpraySQ(self, method=1):
    return self.calculate_Spray_scores(method)
  
  def CalculateMISQ(self):
    self.curr_matrixA, self.curr_matrixB, self.curr_matrixC = self.calculate_matrix()
    return self.calculate_MI_scores(2)

  def calculate_Spray_scores(self, method=1):
    if method == 1:
      #calcullate spray effcet
      spray_effect = 0
      curr_trace_set = self.curr_trace_set.copy()
      pollution_distribute = self.pollution_distribute.copy()
      spray_done = np.ones((self.agent_number,pollution_distribute.shape[0],pollution_distribute.shape[1]))
      for j in range(self.time):  
        spray_area = []
        for i in range(self.agent_number):
          if self.policy_matrix[i, j, 2] == 1:
            r0 = curr_trace_set[i, j, 0]
            c0 = curr_trace_set[i, j, 1]
            for a in range(5):
              for b in range(5):
                r = int(r0 - 2 + a) 
                c = int(c0 - 2 + b)
                effect_rate = 1
                # 判断该区域是否为污染源区域
                # self.Setting.sources
                for n in range(len(self.Setting.sources)):
                  if r == self.Setting.sources[n][0] and c == self.Setting.sources[n][1]:
                    effect_rate = 0.5
                    break
                
                if r >= 0 and r < self.map_shape[0] and c >= 0 and c < self.map_shape[1]:
                  if a == 2 and b == 2:
                    spray_effect = spray_effect + spray_done[i,r,c]*0.15*(self.Setting.sche_step+18-j)*calculate_effect(pollution_distribute[r,c])
                    # spray_effect = spray_effect + calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - effect_rate*calculate_effect(pollution_distribute[r,c])
                    for m in range(self.agent_number):
                      if m != i and spray_done[i,r,c] == 1:
                        spray_done[m,r,c] = 0.1
                  elif (a - 2)**2 + (b - 2)**2 <= 2:
                    spray_effect = spray_effect + spray_done[i,r,c]*0.7*0.15*(self.Setting.sche_step+18-j)*calculate_effect(pollution_distribute[r,c])
                    # spray_effect = spray_effect + 0.5*calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - 0.7*effect_rate*calculate_effect(pollution_distribute[r,c])
                    for m in range(self.agent_number):
                      if m != i and spray_done[i,r,c] == 1:
                        spray_done[m,r,c] = 0.3
                  else:
                    spray_effect = spray_effect + spray_done[i,r,c]*0.5*0.15*(self.Setting.sche_step+18-j)*calculate_effect(pollution_distribute[r,c])
                    # spray_effect = spray_effect + 0.5*calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - 0.5*effect_rate*calculate_effect(pollution_distribute[r,c])
                    for m in range(self.agent_number):
                      if m != i and spray_done[i,r,c] == 1:
                        spray_done[m,r,c] = 0.4
                    
      return spray_effect
    elif method == 2:
      #calcullate spray effcet
      spray_effect = 0
      curr_trace_set = self.curr_trace_set.copy()
      pollution_distribute = self.pollution_distribute.copy()
      spray_done = np.ones((self.agent_number,pollution_distribute.shape[0],pollution_distribute.shape[1]))
      for j in range(self.time):  
        spray_area = []
        for i in range(self.agent_number):
          if self.policy_matrix[i, j, 2] == 1:
            r0 = curr_trace_set[i, j, 0]
            c0 = curr_trace_set[i, j, 1]
            for a in range(5):
              for b in range(5):
                r = int(r0 - 2 + a) 
                c = int(c0 - 2 + b)
                effect_rate = 1
                # 判断该区域是否为污染源区域
                # self.Setting.sources
                for n in range(len(self.Setting.sources)):
                  if r == self.Setting.sources[n][0] and c == self.Setting.sources[n][1]:
                    effect_rate = 0.5
                    break
                
                if r >= 0 and r < self.map_shape[0] and c >= 0 and c < self.map_shape[1]:
                  if a == 2 and b == 2:
                    spray_effect = spray_effect + spray_done[i,r,c]*0.15*(self.Setting.sche_step+18-j)*calculate_effect(pollution_distribute[r,c])
                    # spray_effect = spray_effect + calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - effect_rate*calculate_effect(pollution_distribute[r,c])
                    for m in range(self.agent_number):
                      if m != i and spray_done[i,r,c] == 1:
                        spray_done[m,r,c] = 0.1
                  elif (a - 2)**2 + (b - 2)**2 <= 2:
                    spray_effect = spray_effect + spray_done[i,r,c]*0.7*0.15*(self.Setting.sche_step+18-j)*calculate_effect(pollution_distribute[r,c])
                    # spray_effect = spray_effect + 0.5*calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - 0.7*effect_rate*calculate_effect(pollution_distribute[r,c])
                    for m in range(self.agent_number):
                      if m != i and spray_done[i,r,c] == 1:
                        spray_done[m,r,c] = 0.3
                  else:
                    spray_effect = spray_effect + spray_done[i,r,c]*0.5*0.15*(self.Setting.sche_step+18-j)*calculate_effect(pollution_distribute[r,c])
                    # spray_effect = spray_effect + 0.5*calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - 0.5*effect_rate*calculate_effect(pollution_distribute[r,c])
                    for m in range(self.agent_number):
                      if m != i and spray_done[i,r,c] == 1:
                        spray_done[m,r,c] = 0.4      
          if self.policy_matrix[i, j, 2] == -1:
            for m in range(pollution_distribute.shape[0]):
              for n in range(pollution_distribute.shape[1]):
                if spray_done[i,m,n] == 1:     
                  spray_done[:,m,n] = 1
      return spray_effect
  
  def calculate_Sprayscores_foreveryvehicle(self,method = 1):
    if method == 1:
      #calcullate spray effcet
      spray_effect = 0
      curr_trace_set = self.curr_trace_set.copy()
      pollution_distribute = self.pollution_distribute.copy()
    
      spray_time = np.zeros(self.agent_number)
      spray_effect = np.zeros(self.agent_number)
      spray_effect0 = 0
      for j in range(8):
        for i in range(self.agent_number):
          if self.policy_matrix[i, j, 2] == 1:
            r0 = curr_trace_set[i, j, 0]
            c0 = curr_trace_set[i, j, 1]
            spray_time[i] = spray_time[i] + 1
            for a in range(3):
              for b in range(3):
                r = int(r0 - 1 + a)
                c = int(c0 - 1 + b)
                if r >= 0 and r < self.map_shape[0] and c >= 0 and c < self.map_shape[1]:
                  if a == 1 and b == 1:
                    spray_effect[i] = spray_effect[i] + (0.9)**j*calculate_effect(pollution_distribute[r,c])
                    spray_effect0 = spray_effect0 + (0.9)**j*calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - calculate_effect(pollution_distribute[r,c])
                    # spray_effect[i] = spray_effect[i] + (0.9)**j*0.2*pollution_distribute[r,c]
                    # spray_effect0 = spray_effect0 + (0.9)**j*0.2*pollution_distribute[r,c]
                    # pollution_distribute[r,c] = pollution_distribute[r,c] - 0.2*pollution_distribute[r,c]
                  else:
                    spray_effect[i] = spray_effect[i] + 0.5*(0.9)**j*calculate_effect(pollution_distribute[r,c])
                    spray_effect0 = spray_effect0 + 0.5*(0.9)**j *calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - 0.5*calculate_effect(pollution_distribute[r,c])
                    # spray_effect[i] = spray_effect[i] + 0.5*(0.9)**j*0.15*pollution_distribute[r,c]
                    # spray_effect0 = spray_effect0 + 0.5*(0.9)**j *0.15*pollution_distribute[r,c]
                    # pollution_distribute[r,c] = pollution_distribute[r,c] - 0.15*pollution_distribute[r,c]
      for i in range(self.agent_number):
        if spray_time[i] > 0:
          spray_effect[i] = spray_effect[i] / spray_time[i] 
      return spray_effect0, spray_effect
    elif method == 2:
      #calcullate spray effcet
      spray_effect = 0
      curr_trace_set = self.curr_trace_set.copy()
      pollution_distribute = self.pollution_distribute.copy()
    
      spray_time = np.zeros(self.agent_number)
      spray_effect = np.zeros(self.agent_number)
      spray_effect0 = 0
      for j in range(8):
        for i in range(self.agent_number):
          if self.policy_matrix[i, j, 2] == 1:
            r0 = curr_trace_set[i, j, 0]
            c0 = curr_trace_set[i, j, 1]
            spray_time[i] = spray_time[i] + 1
            for a in range(3):
              for b in range(3):
                r = int(r0 - 1 + a)
                c = int(c0 - 1 + b)
                if r >= 0 and r < self.map_shape[0] and c >= 0 and c < self.map_shape[1]:
                  if a == 1 and b == 1:
                    spray_effect[i] = spray_effect[i] + (0.8+0.2*np.max((self.MIforeverypoint[r,c]+0.4,1)))**j*calculate_effect(pollution_distribute[r,c])
                    spray_effect0 = spray_effect0 + (0.8+0.2*np.max((self.MIforeverypoint[r,c]+0.4,1)))**j*calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - calculate_effect(pollution_distribute[r,c])
                  else:
                    spray_effect[i] = spray_effect[i] + 0.5*(0.8+0.2*np.max((self.MIforeverypoint[r,c]+0.4,1)))**j*calculate_effect(pollution_distribute[r,c])
                    spray_effect0 = spray_effect0 + 0.5*(0.8+0.2*np.max((self.MIforeverypoint[r,c]+0.4,1)))**j *calculate_effect(pollution_distribute[r,c])
                    pollution_distribute[r,c] = pollution_distribute[r,c] - 0.5*calculate_effect(pollution_distribute[r,c])
      for i in range(self.agent_number):
        if spray_time[i] > 0:
          spray_effect[i] = spray_effect[i] / spray_time[i] 
      return spray_effect0, spray_effect

  def calculate_MI_scores(self, method = 2):
    #calculate mi about selected point to all points at one time
    if method == 1:
      #calculate mi at one time point
      curr_matrixB = self.curr_matrixB
      allpoint_list = []
      for i in range (self.Setting.task_extent[0],self.Setting.task_extent[1]):
          for j in range (self.Setting.task_extent[2],self.Setting.task_extent[3]):
              allpoint_list.append([i, j, self.model.time_stamp])
      allpoint = np.array(allpoint_list)
      
      processed_points = np.unique(curr_matrixB, axis=0)
      train_data = self.model.get_data_x()
      
      nrows, ncols = train_data.shape
      dtype={'names':['f{}'.format(i) for i in range(ncols)],
          'formats':ncols * [train_data.dtype]}
      mid_points = np.intersect1d(train_data.view(dtype), processed_points.view(dtype))
      processed_points2 = np.setdiff1d(processed_points.view(dtype), mid_points)
      processed_points2 = processed_points2.view(train_data.dtype).reshape(-1, ncols)
      self.model.add_data_x(processed_points2)
      _, _, prior_cov, poste_cov = self.model.prior_poste(allpoint)
      if processed_points2.shape[0] > 0:
          self.model.reduce_data_x(processed_points2.shape[0])
      # prior_entropy = gaussian_entropy_multivariate(prior_cov)
      # poste_entropy = gaussian_entropy_multivariate(poste_cov)
      # mi = prior_entropy - poste_entropy
      mi = (prior_cov.trace()- poste_cov.trace())/prior_cov.shape[0]
      # print(mi)
    elif method == 2:
      #calculate mi at whole time period，all_state are about the whole time
      curr_matrixC = self.curr_matrixC
      allpoint = self.allpoint
      processed_points = np.unique(curr_matrixC, axis=0)
      # print(processed_points)
      train_data = self.model.get_data_x()
      
      nrows, ncols = train_data.shape
      dtype={'names':['f{}'.format(i) for i in range(ncols)],
          'formats':ncols * [train_data.dtype]}
      mid_points = np.intersect1d(train_data.view(dtype), processed_points.view(dtype))
      processed_points2 = np.setdiff1d(processed_points.view(dtype), mid_points)
      processed_points2 = processed_points2.view(train_data.dtype).reshape(-1, ncols)
      self.model.add_data_x(processed_points2)
      _, _, prior_cov, poste_cov = self.model.prior_poste(allpoint)
      if processed_points2.shape[0] > 0:
          self.model.reduce_data_x(processed_points2.shape[0])
      # prior_entropy = gaussian_entropy_multivariate(prior_cov)
      # poste_entropy = gaussian_entropy_multivariate(poste_cov)
      # mi = prior_entropy - poste_entropy
      mi = (prior_cov.trace()- poste_cov.trace())/prior_cov.shape[0]
        
    return mi
  
  def calculate_wolume_scores(self):
    spray_time = 0
    for j in range(self.time):
      for i in range(self.agent_number):
        if self.policy_matrix[i, j, 2] == 1:
          spray_time = spray_time + 1 * (0.95)**j
    return spray_time

  def calculate_matrix(self):
    #calculate three matrix, one for spray effect, one for mi and one as before
    #first as before
    matrixA =  np.zeros(self.map_shape)
    for i in range(self.agent_number):
      for j in range(self.time + 1):
        if j == 0:
          continue
        x = self.curr_trace_set[i, j, 0:2][0].astype(np.int16)
        y = self.curr_trace_set[i, j, 0:2][1].astype(np.int16)
        matrixA[x, y] += 1
          
    #second matrix for mi calculate at one times
    num = 0
    matrixB = np.zeros(((self.time+1)*self.agent_number,3))
    for x in range(self.map_shape[0]):
      for y in range(self.map_shape[1]):
        if matrixA[x, y] > 0.5:
          matrixB[num,0] = x
          matrixB[num,1] = y
          matrixB[num,2] = self.model.time_stamp
          num = num + 1
    matrixB = matrixB[0:num]
    
    #third matrix for mi calculate at different times
    num = 0
    matrixC = np.zeros(((self.time+1)*self.agent_number,3))
    mid = np.zeros((1,3))
    for j in range(self.time + 1):
      for i in range(self.agent_number):
        if j == 0:
          continue
        x = self.curr_trace_set[i, j, 0:2][0].astype(np.int16)
        y = self.curr_trace_set[i, j, 0:2][1].astype(np.int16)
        z = self.curr_trace_set[i, j, 3]
        can = np.array([[x,y,z]])
        if np.all(np.any(can == mid, axis=1)):
          continue
        mid = np.vstack((mid,can))
        matrixC[num,0] = x
        matrixC[num,1] = y
        matrixC[num,2] = z
        num = num + 1
      mid = np.zeros((1,3))
    matrixC = matrixC[0:num]
    
    return matrixA, matrixB, matrixC
  
  def calculate_trace_set(self):
    curr_trace_set =  np.zeros((self.agent_number, self.time + 1, 4))
    for i in range(self.agent_number):
      for j in range(self.time + 1):
        if(j == 0):
          curr_trace_set[i, j, 0:2] = np.array(self.agent_init_position[i])
          curr_trace_set[i, j, 2:3] = self.Setting.water_volume
          curr_trace_set[i, j, 3] = self.model.time_stamp
        else:
          # curr_trace_set[i, j, 0:2] = np.array(self.move_matrix[self.policy_matrix[i, j - 1]][0:2]) + curr_trace_set[i, j - 1, 0:2]
          curr_trace_set[i, j, 0:2] = np.array(self.policy_matrix[i, j - 1, 0:2]) + curr_trace_set[i, j - 1, 0:2]
          # 洒水
          # if self.move_matrix[self.policy_matrix[i, j - 1]] == 1:
          if self.policy_matrix[i, j - 1, 2] == 1:
            # curr_trace_set[i, j, 2] = curr_trace_set[i, j - 1, 2] - self.move_matrix[self.policy_matrix[i, j - 1]][2]
            curr_trace_set[i, j, 2] = curr_trace_set[i, j - 1, 2] - self.policy_matrix[i, j - 1, 2]
          # 补水
          elif self.policy_matrix[i, j - 1, 2] == -1:
            curr_trace_set[i, j, 2] = curr_trace_set[i, j - 1, 2] - self.policy_matrix[i, j - 1, 2] * self.Setting.replenish_speed
            if curr_trace_set[i, j, 2] >= self.Setting.water_volume:
              curr_trace_set[i, j, 2] = self.Setting.water_volume
          # 其他
          else:
            curr_trace_set[i, j, 2] = curr_trace_set[i, j - 1, 2]
          curr_trace_set[i, j, 3] = self.model.time_stamp + self.time_co * j
          
    return curr_trace_set

  def GetAgentNumber(self) -> int:
    return self.agent_number

  def GetMaxTime(self) -> int:
    return self.time
  
  def GetSprayTime(self,agent_number) -> int:
    num = 0
    for i in range(self.time):
      if self.policy_matrix[agent_number, i, 2] == 1:
        num = num + 1
    return num
  
  def GetDontSprayTime(self,agent_number) -> int:
    num = 0
    for i in range(self.time):
      if self.policy_matrix[agent_number, i, 2] == 0:
        num = num + 1
    return num
  
  def GetCurrentInfo(self):
    allpoint_list = []
    for i in range(self.map_shape[0]):
      for j in range(self.map_shape[1]):
        allpoint_list.append([i, j, self.model.time_stamp])
    allpoint = np.array(allpoint_list)
    _, _, prior_cov, poste_cov = self.model.prior_poste(allpoint)
    mi = (prior_cov.trace()- poste_cov.trace())/prior_cov.shape[0]
    return mi

  def GetPossibleActions(self) -> int:
    return self.possible_actions
  
  def GetAgentInitialPosition(self):
    return self.agent_init_position

  def GetMoveMatrices(self):
    return self.move_matrix

  def GetMapShape(self):
    return self.map_shape
  
  def GetLastInfo(self):
    return self.last_info
  
  def UpdateInfo(self):
    self.last_info = self.GetCurrentInfo()

  def CheckValid(self, agent_position_list):
    # 判断位置是否超限
    if(np.max(agent_position_list[:, 0:2] <= -1) == True):
      return False
    if(np.max(agent_position_list[:, 0] >= self.map_shape[0]) == True):
      return False
    if(np.max(agent_position_list[:, 1] >= self.map_shape[1]) == True):
      return False
    # 判断洒水是否超限
    if(np.max(agent_position_list[:, 2] >= self.Setting.water_volume + 1) == True):
      return False
    if(np.max(agent_position_list[:, 2] <= -1) == True):
      return False 
    return True
  
  def adaptive_update(self, model, pollution_distribute, allpoint, Setting):
    # 初始智能体位置
    for i in range(self.agent_init_position.shape[0]):
      self.agent_init_position[i] = self.curr_trace_set[i,self.Setting.adaptive_step, 0:2]

    # 预测模型及所需参数
    self.Setting = Setting
    if Setting.current_step > Setting.max_num_samples - Setting.sche_step:
      if Setting.max_num_samples - Setting.current_step > 7:
        self.Setting.sche_step = Setting.max_num_samples - Setting.current_step
      else:
        self.Setting.sche_step = 8
    else:
      self.Setting.sche_step = Setting.sche_step
      
    self.model = model
    self.pollution_distribute = pollution_distribute
    self.allpoint = allpoint
    
    # 智能体策略矩阵,更新策略矩阵
    for i in range(self.time - self.Setting.adaptive_step):
      self.policy_matrix[:,i,:] = self.policy_matrix[:,i+self.Setting.adaptive_step,:]
      self.curr_trace_set[:,i,:] = self.curr_trace_set[:,i+self.Setting.adaptive_step,:] 
    self.curr_trace_set[:,self.time - self.Setting.adaptive_step,:] =  self.curr_trace_set[:,self.time,:]
    
    # 根据现在的长度更新
    residue_length = self.time - self.Setting.adaptive_step
    self.time = self.Setting.sche_step
    self.policy_matrix = self.policy_matrix[:,0:self.time,:]
    self.curr_trace_set = self.curr_trace_set[:,0:(self.time+1),:]
    adaptive_step = self.time - residue_length
    
    # 补充策略矩阵中缺失的部分
    for i in range(adaptive_step):
      for j in range(self.agent_number):
        if self.curr_trace_set[j,self.time - adaptive_step + i, 2] >= 1 and self.policy_matrix[j,self.time - adaptive_step - 1 + i, 2] != -1:
          self.policy_matrix[j, self.time - adaptive_step + i, 2] = 1
          self.policy_matrix[j, self.time - adaptive_step + i, 0] = 0
          self.policy_matrix[j, self.time - adaptive_step + i, 1] = 0
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 0] = self.curr_trace_set[j,self.time - adaptive_step + i, 0]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 1] = self.curr_trace_set[j,self.time - adaptive_step + i, 1]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 2] = self.curr_trace_set[j,self.time - adaptive_step + i, 2] - 1
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 3] = self.curr_trace_set[j,self.time - adaptive_step + i, 3] + self.time_co
        elif self.curr_trace_set[j,self.time - adaptive_step + i, 2] < 1:
          self.policy_matrix[j, self.time - adaptive_step + i, 2] = -1
          self.policy_matrix[j, self.time - adaptive_step + i, 0] = 0
          self.policy_matrix[j, self.time - adaptive_step + i, 1] = 0
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 0] = self.curr_trace_set[j,self.time - adaptive_step + i, 0]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 1] = self.curr_trace_set[j,self.time - adaptive_step + i, 1]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 2] \
            = self.curr_trace_set[j,self.time - adaptive_step + i, 2] + self.Setting.replenish_speed
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 3] = self.curr_trace_set[j,self.time - adaptive_step + i, 3] + self.time_co
        elif self.curr_trace_set[j,self.time - adaptive_step + i, 2] >= self.Setting.water_volume:
          self.policy_matrix[j, self.time - adaptive_step + i, 2] = 1
          self.policy_matrix[j, self.time - adaptive_step + i, 0] = 0
          self.policy_matrix[j, self.time - adaptive_step + i, 1] = 0
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 0] = self.curr_trace_set[j,self.time - adaptive_step + i, 0]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 1] = self.curr_trace_set[j,self.time - adaptive_step + i, 1]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 2] = self.curr_trace_set[j,self.time - adaptive_step + i, 2] - 1
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 3] = self.curr_trace_set[j,self.time - self.Setting.adaptive_step + i, 3] + self.time_co
        elif self.curr_trace_set[j,self.time - adaptive_step + i, 2] < self.Setting.water_volume \
          and self.policy_matrix[j,self.time - adaptive_step - 1 + i, 2] == -1:
          self.policy_matrix[j, self.time - adaptive_step + i, 2] = -1
          self.policy_matrix[j, self.time - adaptive_step + i, 0] = 0
          self.policy_matrix[j, self.time - adaptive_step + i, 1] = 0
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 0] = self.curr_trace_set[j,self.time - adaptive_step + i, 0]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 1] = self.curr_trace_set[j,self.time - adaptive_step + i, 1]
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 2] \
            = self.curr_trace_set[j,self.time - adaptive_step + i, 2] + self.Setting.replenish_speed
          self.curr_trace_set[j,self.time - adaptive_step + i + 1, 3] = self.curr_trace_set[j,self.time - adaptive_step + i, 3] + self.time_co
    
    #当前轨迹所覆盖矩阵
    self.curr_matrixA, self.curr_matrixB, self.curr_matrixC = self.calculate_matrix()

    
    #计算时刻已观测点对当前层各点的信息量
    # self.MIforeverypoint
    allpoint_list = []
    for i in range (self.Setting.task_extent[0],self.Setting.task_extent[1]):
        for j in range (self.Setting.task_extent[2],self.Setting.task_extent[3]):
            allpoint_list.append([i, j, self.model.time_stamp])
    point = np.array(allpoint_list)
    prior_diag_std, poste_diag_std, _, _ = self.model.prior_poste(point)
    hprior = gaussian_entropy(prior_diag_std.ravel())
    hposterior = gaussian_entropy(poste_diag_std.ravel())
    mi_all = hprior - hposterior
    if np.any(mi_all < 0.0):
        print(mi_all.ravel())
        raise ValueError("Predictive MI < 0.0!")
    normed_mi = (mi_all - mi_all.min()) / mi_all.ptp()
    MI_information = np.zeros((self.Setting.task_extent[1]-self.Setting.task_extent[0],self.Setting.task_extent[3]-self.Setting.task_extent[2]))
    for i in range (self.Setting.task_extent[0],self.Setting.task_extent[1]):
        for j in range (self.Setting.task_extent[2],self.Setting.task_extent[3]):
            MI_information[i,j] = normed_mi[i*(self.Setting.task_extent[3]-self.Setting.task_extent[2])+j]
    self.MIforeverypoint = MI_information