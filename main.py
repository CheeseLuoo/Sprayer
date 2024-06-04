from pathlib import Path
import numpy as np
import pandas as pd
import os
import time as tm
import Sprayer_PDE as SP
import Sprinkler_Scheduling

def get_multi_robots(Setting):
    vehicle_team = dict()
    for i in range(Setting.team_size):
        robot = Sprinkler_Scheduling.robots.SPRINKLER_REPLENISHANYWHERE(
            init_state = np.array([Setting.x_init[-1, 0], Setting.x_init[-1, 1]]),
            Setting = Setting
        )
        vehicle_team[i+1] = robot
    return vehicle_team


def get_strategy(rng, Setting, vehicle_team):
    if Setting.strategy_name == "EffectOrientedSelectiveSpray":
        strategy = Sprinkler_Scheduling.strategies.SAEffectOrientedSelectiveSpray(
                task_extent=Setting.task_extent,
                rng=rng,
                vehicle_team=vehicle_team,
            )
    elif Setting.strategy_name == "EffectOrientedNonDecoupledSpray":
        strategy = Sprinkler_Scheduling.strategies.SAEffectOrientedNonDecoupledSpray2(
                task_extent=Setting.task_extent,
                rng=rng,
                vehicle_team=vehicle_team,
            )
    elif Setting.strategy_name == "MaximumCoverageSpray":
        strategy = Sprinkler_Scheduling.strategies.SAMaximumCoverageSpray(
                task_extent=Setting.task_extent,
                rng=rng,
                vehicle_team=vehicle_team,
            )
    elif Setting.strategy_name == "NoSpray":
        strategy = Sprinkler_Scheduling.strategies.NoSpray(
                task_extent=Setting.task_extent,
                rng=rng,
                vehicle_team=vehicle_team,
            )
    elif Setting.strategy_name == "EffectOrientedMCTSSpray":
        strategy = Sprinkler_Scheduling.strategies.MCTSSpray(
                task_extent=Setting.task_extent,
                rng=rng,
                vehicle_team=vehicle_team,
            )
    return strategy


def get_evaluator():
    evaluator = Sprinkler_Scheduling.experiments.Evaluator()
    return evaluator

def get_env_model(Setting):
    model = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                    initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = 0) # build model
    return model

def get_gprmodel(Setting, y_init, kernel):
    model = Sprinkler_Scheduling.models.GPR(
        x_train=Setting.x_init,
        y_train=y_init,
        kernel=kernel,
        noise=Setting.init_noise,
        lr_hyper=Setting.lr_hyper,
        lr_nn=Setting.lr_nn,
        is_normalized = True,
        time_stamp = Setting.time_stamp,
    )
    return model


def run(rng, model, Setting, sensor, evaluator, logger, vehicle_team) -> None:
    current_step = 0 
    adaptive_step = Setting.adaptive_step 
    change_step = 0
    spray_effect = 0 
    result, MI_information, observed_env, computed_effect = None, None, None, None
    while current_step < Setting.max_num_samples:
        
        allpoint_list = []
        env_list = []
        for i in range (Setting.task_extent[0],Setting.task_extent[1]):
            for j in range (Setting.task_extent[2],Setting.task_extent[3]):
                allpoint_list.append([i, j, model.time_stamp])
                env_list.append(Setting.env[i,j])
        allpoint = np.array(allpoint_list)
        env = np.array(env_list)
        mean, _ = model(allpoint)
        sprayeffect_all = Sprinkler_Scheduling.objectives.sprayeffect.spray_effect(allpoint, allpoint, mean, Setting.task_extent).ravel()
        prior_diag_std, poste_diag_std, _, _ = model.prior_poste(allpoint)
        hprior = Sprinkler_Scheduling.objectives.entropy.gaussian_entropy(prior_diag_std.ravel())
        hposterior = Sprinkler_Scheduling.objectives.entropy.gaussian_entropy(poste_diag_std.ravel())
        mi_all = hprior - hposterior
        if np.any(mi_all < 0.0):
            print(mi_all.ravel())
            raise ValueError("Predictive MI < 0.0!")
        
        sprayeffect_all = Sprinkler_Scheduling.objectives.sprayeffect.spray_effect(allpoint, allpoint, env, Setting.task_extent).ravel()
        MI_information = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        observed_env = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        computed_effect = np.zeros((Setting.task_extent[1]-Setting.task_extent[0],Setting.task_extent[3]-Setting.task_extent[2]))
        for i in range (Setting.task_extent[0],Setting.task_extent[1]):
            for j in range (Setting.task_extent[2],Setting.task_extent[3]):
                MI_information[i,j] = mi_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
                observed_env[i,j] = Setting.env[i,j]
                computed_effect[i,j] = sprayeffect_all[i*(Setting.task_extent[3]-Setting.task_extent[2])+j]
                
        Setting.current_step = current_step
        
        # scheduling and update agent goals ###################################################
        if adaptive_step >= Setting.adaptive_step:
            start = tm.time()
            result = Setting.strategy.get(model = model, Setting = Setting, pred = observed_env)
            adaptive_step = 0
            for id, vehicle in vehicle_team.items():
                vehicle.set_goals(result[id][0],result[id][1])
            end = tm.time()
            print('search_time')
            print(end-start)    
            
        # calculate metrix and save 
        coverage, mean_airpollution, max_airpollution = evaluator.eval_results(Setting.env, Setting.task_extent, vehicle_team)
        logger.append(current_step, Setting.env, observed_env, MI_information, computed_effect, vehicle_team, coverage, mean_airpollution, max_airpollution, spray_effect)
           
        # change source,
        if change_step >= Setting.R_change_interval:
            Setting.R =  -3 * np.ones((Setting.grid_x, Setting.grid_y)) + 6 * rng.random((Setting.grid_x, Setting.grid_y))
            change_step = 0
            if Setting.randomsource == True:
                # gengerate two set of random numbers for source locations
                numbers = rng.randint(0, 4, size=Setting.sourcenum * 2)
                pairs = rng.choice(numbers, size=(Setting.sourcenum, 2), replace=False)
                for i in range(Setting.sourcenum):
                    number = rng.randint(50, 70, size=1)
                    if Setting.RR[i,0]+pairs[i,0]-2 < Setting.grid_x-1 and Setting.RR[i,0] + pairs[i,0] - 2 >=0:
                        Setting.RR[i,0] = int(Setting.RR[i,0]+pairs[i,0]-2)
                    if Setting.RR[i,1]+pairs[i,1]-2 < Setting.grid_y-1 and Setting.RR[i,1] + pairs[i,1] - 2 >=0:
                        Setting.RR[i,1] = int(Setting.RR[i,1]+pairs[i,1]-2)
                    Setting.RR[i,2] = number
                tstart = current_step

        s = 1
        for i in range(Setting.sourcenum):
             Setting.R[Setting.RR[i,0],Setting.RR[i,1]] = s*Setting.RR[i,2]
        
        env_model1 = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                initial_field = Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = current_step * Setting.delta_t)
        env_withoutspray = env_model1.solve(Setting.delta_t)[-1]
            
        # update state 
        x_new = []
        y_new = []
        for id, vehicle in vehicle_team.items():
            vehicle.update()
            current_state = vehicle.state.copy().reshape(1, -1)
            x_new.append(current_state)
            y_new.append(sensor.sense(current_state, rng).reshape(-1, 1))
            if Setting.current_step == 0:
                Setting.data_sprayer_train.append(pd.DataFrame())
            if vehicle.spray_flag == True:
                new_pd = pd.DataFrame({"time":(Setting.current_step + 1) * Setting.delta_t, "x":current_state[0,0],\
                                        "y":current_state[0,1], "spray_volume":200},index=[0])
                Setting.data_sprayer_train[id-1] = pd.concat([Setting.data_sprayer_train[id-1],new_pd])
            else:
                new_pd = pd.DataFrame({"time":(Setting.current_step + 1) * Setting.delta_t, "x":current_state[0,0],\
                                        "y":current_state[0,1], "spray_volume":0},index=[0])
                Setting.data_sprayer_train[id-1] = pd.concat([Setting.data_sprayer_train[id-1],new_pd])    
        
        env_model2 = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = current_step * Setting.delta_t) # build model
        env_withspray = env_model2.solve(Setting.delta_t)[-1]
        
        for id, vehicle in vehicle_team.items():
            current_state = vehicle.state.copy().reshape(1, -1).astype(int)
            for i in range(Setting.sourcenum):
                if ((current_state[0,0]-Setting.RR[i,0])**2 + (current_state[0,1]-Setting.RR[i,1])**2) <= 2:
                    in_flag = False
                    for j in range(len(Setting.sources)):
                        if Setting.RR[i,0] == Setting.sources[j][0] and Setting.RR[i,1] == Setting.sources[j][1]:
                            in_flag = True
                    if in_flag:
                        continue
                    else:
                        Setting.sources.append([Setting.RR[i,0], Setting.RR[i,1]])         
                        
        Setting.env = env_withspray
        sensor.set_env(Setting.env)
        for i in range(len(Setting.sources)-1, -1, -1):
            if Setting.env[Setting.sources[i][0],Setting.sources[i][1]] <= 45:
                del Setting.sources[i]    
        
        spray_effect = np.sum(env_withoutspray - Setting.env)
        # print(spray_effect)
            
        # using new data to update gpr model
        x_new = np.concatenate(x_new, axis=0)
        y_new = np.concatenate(y_new, axis=0)
        #add time dim
        model.time_stamp = model.time_stamp + Setting.time_co
        Setting.time_stamp = model.time_stamp
        model_input = np.zeros((x_new.shape[0],3))
        model_input[:,0:2] = x_new
        model_input[:,2:3] = model.time_stamp
        #optimize model
        model.add_data(model_input, y_new)
        model.optimize(num_iter=len(y_new), verbose=False)
        
        adaptive_step = adaptive_step + 1     
        current_step = current_step + 1
        change_step = change_step + 1  
    return 0

def Set_initual_data(rng,Setting,sensor):
    if Setting.randomsource == True:
        # gengerate two set of random numbers for source locations
        numbers = rng.randint(0, 19, size=Setting.sourcenum * 2)
        pairs = rng.choice(numbers, size=(Setting.sourcenum, 2), replace=False)
        for i in range(Setting.sourcenum):
            number = rng.randint(50, 70, size=1)
            Setting.RR[i,0] = int(pairs[i,0])
            Setting.RR[i,1] = int(pairs[i,1])
            Setting.RR[i,2] = number

    s = 1
    Setting.R =  -3 * np.ones((Setting.grid_x, Setting.grid_y)) + 6 * rng.random((Setting.grid_x, Setting.grid_y))
    for i in range(Setting.sourcenum):
         Setting.R[Setting.RR[i,0],Setting.RR[i,1]] = s*Setting.RR[i,2]
            
    env_model = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                 initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = 0) # build model
    Setting.env = env_model.solve(10)[-1]
    sensor.set_env(Setting.env)
    
    env_model = SP.Diffusion_Model(x_range = Setting.grid_x, y_range = Setting.grid_y,\
                    initial_field =  Setting.env, R_field =  Setting.R, data_sprayer_train = Setting.data_sprayer_train, t_start = 0) # build model

    y_init = np.zeros((Setting.num_init_samples,1))
    y_stations = np.zeros((Setting.station_size*Setting.time_before_sche,1))
    time_init = np.zeros((Setting.num_init_samples,1))
    time_stations = np.zeros((Setting.station_size*Setting.time_before_sche,1))

    for time in range(Setting.time_before_sche):
        time_stations[Setting.station_size*time:Setting.station_size*(time+1)] = (time-Setting.time_before_sche+1)*1
        Setting.env = env_model.solve((time+1)*5)[-1]
        sensor.set_env(Setting.env)
        
    for time in range(Setting.num_init_samples):
        y_init[time] = sensor.sense(states=Setting.x_init[time], rng=rng).reshape(-1, 1)
        if time == 0:
            y_stations[:] = y_init[time] - 20
        time_init[time] = (time+1)*1
        Setting.env = env_model.solve((1+Setting.time_before_sche+time)*5)[-1]
        sensor.set_env(Setting.env)
        
    Setting.x_init = np.hstack((Setting.x_init,time_init))

    Setting.x_stations = Setting.x_station
    for i in range(Setting.time_before_sche-1):
        Setting.x_stations = np.vstack((Setting.x_stations,Setting.x_station))
    Setting.x_stations = np.hstack((Setting.x_stations,time_stations))

    Setting.x_init = np.vstack((Setting.x_stations,Setting.x_init))
    return np.vstack((y_stations,y_init))

def main():
    args = Sprinkler_Scheduling.experiments.argparser.parse_arguments()
    
    Setting = Sprinkler_Scheduling.utilities.Config(root_dir = args.root_dir, save_name = args.save_name,
                diffusivity_K = args.diffusivity_K, grid_x = args.grid_x, grid_y = args.grid_y, time_co = args.time_co, delta_t = args.delta_t,
                sensing_rate = args.sensing_rate, noise_scale = args.noise_scale, num_init_samples = args.num_init_samples, seed = args.seed,
                time_before_sche = args.time_before_sche, sourcenum = args.sourcenum, R_change_interval = args.R_change_interval,
                init_amplitude = args.amplitude, init_lengthscale = args.lengthscale, init_noise = args.init_noise,
                lr_hyper = args.lr_hyper, lr_nn = args.lr_nn,
                team_size = args.team_size, water_volume = args.water_volume, replenish_speed = args.replenish_speed,
                max_num_samples = args.max_num_samples ,bound1 = args.bound1, bound2 = args.bound2, bound3 = args.bound3,
                alpha = args.alpha,
                Strategy_Name = args.strategy_name,
                sche_step = args.sche_step, adaptive_step = args.adaptive_step, Env = args.Env,
                effect_threshold = args.effect_threshold)

    # environment initual
    # env_model = get_env_model(Setting)
    # Setting.env = env_model.solve(Setting.delta_t)
    
    # save directory
    # starttime = Setting.starttime.replace(' ', '-').replace(':', '-')
    # Setting.save_dir = '{}/{}/teamsize_{}'.format(Setting.root_dir, Setting.strategy_name, Setting.team_size)
    # Setting.save_dir = '{}/{}/numsource_{}'.format(Setting.root_dir, Setting.strategy_name, Setting.sourcenum)
    Setting.save_dir = '{}/{}/bound1_{}teamsize_{}'.format(Setting.root_dir, Setting.strategy_name, Setting.bound1, Setting.team_size)
    # Setting.save_name = args.save_name
    evaluator = get_evaluator()
    logger = Sprinkler_Scheduling.experiments.Logger(None, Setting)
    
    sensor = Sprinkler_Scheduling.sensors.Sprinkler(Setting = Setting)
    rng = Sprinkler_Scheduling.experiments.utilities.seed_everything(Setting = Setting)
    
    # model
    y_init = Set_initual_data(rng,Setting,sensor)
    Setting.time_stamp = Setting.x_init[:,2].max(axis=0, keepdims=False)
    kernel = Sprinkler_Scheduling.kernels.RBF(Setting)
    model = get_gprmodel(Setting, y_init, kernel)
    model.optimize(num_iter=model.num_train, verbose=True)
    
    # robot
    vehicle_team =  get_multi_robots(Setting)
        
    # strategy
    Setting.strategy = get_strategy(rng, Setting, vehicle_team)

    # experiment search
    start = tm.time()
    run(rng, model, Setting, sensor, evaluator, logger, vehicle_team)
    end = tm.time()
    logger.save(end-start)  # I temporarily removed "makefile()".
    # Sprinkler_Scheduling.experiments.utilities.print_metrics(logger, Setting.max_num_samples-1)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()  # remove parameter for old one