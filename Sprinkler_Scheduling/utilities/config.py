from typing import List

import numpy as np


class Config:
    """Configuring some parameters."""
    def __init__(self, root_dir = "./outputs", save_name = "text", strategy = None, 
                 diffusivity_K =1.2, grid_x = 20, grid_y = 20, time_co = 0.0001, delta_t = 0.01,
                 sensing_rate = 1.0, noise_scale = 1.0, num_init_samples = 1, seed = 11,
                 time_before_sche = 5, station_size = 1, sourcenum = 4, R_change_interval = 50,
                 init_amplitude = 1.0, init_lengthscale = 0.5, init_noise = 1.0,
                 lr_hyper = 0.01, lr_nn = 0.001,
                 team_size = 3, water_volume=4, replenish_speed = 1,
                 max_num_samples = 18, current_step = 0 ,bound1 = 100, bound2 = 15, bound3 = 100,
                 alpha = [0.75,0.9,0.99,1.05,1.5],
                 Strategy_Name = "SA_OnlyonetimeMI_simpleeffect",
                 sche_step = 18, adaptive_step = 3, Env = "Dynamic",
                 effect_threshold = 0.0) -> None:
        
        # self.starttime = '2018-11-23 08:00:00'
        
        self.root_dir = root_dir
        self.save_dir = root_dir
        self.save_name = save_name
        
        # pollution diffusion parameters
        self.diffusivity_K = diffusivity_K # diffusivity，
        self.grid_x = grid_x 
        self.grid_y = grid_y
        # self.env = 80 * np.ones((grid_x, grid_y))\
        #             + 0 * np.random.random((grid_x, grid_y))# randomly initialize "initial_field" map matrix around 250
        self.env = 25 * np.ones((grid_x, grid_y))\
                    + 0 * np.random.random((grid_x, grid_y))
        
        #source
        self.randomsource = True
        self.sourcenum = sourcenum
        # self.sourcenum = team_size
        self.R =  -3 * np.ones((grid_x, grid_y)) + 6 * np.random.random((grid_x, grid_y)) # initialize pollution resource map matrix
        self.R_change_interval = R_change_interval
        self.data_sprayer_train = [] 
        self.RR = np.zeros((self.sourcenum, 3)).astype(int)

        for a in range(self.sourcenum):
            self.R[self.RR[a,0],self.RR[a,1]] = self.RR[a,2]
        self.sources = []

        #time parameter
        self.time_co = 0.1 
        self.delta_t = 10 
        
        #range
        self.env_extent = [0, self.grid_x, 0, self.grid_y]
        self.task_extent = [0, self.grid_x, 0, self.grid_y]
        
        #sensing parameter
        self.sensing_rate = sensing_rate
        self.noise_scale = noise_scale
        
        # experiment parameter
        self.num_init_samples = num_init_samples
        self.seed = seed
        self.max_num_samples = max_num_samples
        self.current_step = current_step
        # bound in SA
        self.bound1 = bound1 
        self.bound2 = bound2 
        self.bound3 = bound3 

        self.alpha = alpha
        self.strategy = strategy #class
        self.strategy_name = Strategy_Name
        self.sche_step = sche_step
        self.adaptive_step = adaptive_step
        if self.adaptive_step > self.sche_step:
            raise ValueError("adaptive_step must smaller than sche_step")
        self.Env = Env
        self.effect_threshold = effect_threshold
        
        self.x_init = np.zeros((self.num_init_samples,2))
        self.x_init[0,0] = 10.0
        self.x_init[0,1] = 10.0

        self.station_size = station_size
        self.x_station = np.zeros((self.station_size,2))
        self.x_station[0,0] = 16.0
        self.x_station[0,1] = 16.0
        self.water_station = np.zeros((1,2))
        self.water_station[0,0] = 5.0
        self.water_station[0,1] = 5.0
        
        self.time_before_sche = time_before_sche
        
        self.amplitude = init_amplitude
        self.lengthscale = init_lengthscale
        self.init_noise = init_noise
        self.time_stamp = 0
        
        self.lr_hyper = lr_hyper
        self.lr_nn = lr_nn
    
        # vehicle team
        self.team_size = team_size
        self.replenish_speed = replenish_speed
        self.water_volume = water_volume
        
        self.accept_rate = []

    
        

