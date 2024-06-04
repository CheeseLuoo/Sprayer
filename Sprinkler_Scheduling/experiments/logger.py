from pathlib import Path
import numpy as np
import pickle
import os
from ..experiments.utilities import makefile

class Logger:
    """Save all the variables for visualization."""
    def __init__(self, args = None, Setting = None) -> None:
        if args != None:
            self.save_data = {'info': {'gridx': args.grid_x,
                                       'gridy': args.grid_y,
                                       'time_co':args.time_co,
                                       'delta_t':args.delta_t,
                                       'strategy_name':args.strategy_name,
                                       'env_type':args.Env,
                                       'effect_threshold':args.effect_threshold,
                                       'team_size':args.team_size,
                                       'sche_step':args.sche_step,
                                       'adaptive_step':args.adaptive_step,
                                       'random_seed':args.seed}, 
                              'time_series': dict(), 
                              'truth_env': [],
                              'observed_env': [],
                              'MI_information': [],
                              'computed_effect': [],
                              'mean_airpollution': [],
                              'max_airpollution': [], 
                              'coverage':[],
                              'spray_effect':[],
                              'runtime': 0.0}
            self.save_dir = args.save_dir
            os.makedirs(args.save_dir, exist_ok=True)
            # self.save_name = f'E{args.env}_D{args.dimension}_FS{args.fire_size}_FN{args.fire_num}_U{args.update_interval}_T{args.horizon}_C{args.communication}_N{args.num_robot}_I{args.image_size}_S{args.suppress_size}_M{args.measure_correct:.2f}_A{args.alpha:.2f}_B{args.beta:.2f}'
            self.save_name = args.save_name
        else:
            self.save_data = {'info': {'gridx': Setting.grid_x,
                                       'gridy': Setting.grid_y,
                                       'time_co':Setting.time_co,
                                       'delta_t':Setting.delta_t,
                                       'strategy_name':Setting.strategy_name,
                                       'env_type':Setting.Env,
                                       'effect_threshold':Setting.effect_threshold,
                                       'team_size':Setting.team_size,
                                       'sche_step':Setting.sche_step,
                                       'adaptive_step':Setting.adaptive_step,
                                       'random_seed':Setting.seed}, 
                              'time_series': dict(), #区分不同agent的数据存放在这里
                              'truth_env': [],
                              'observed_env': [],
                              'MI_information': [],
                              'computed_effect': [],
                              'mean_airpollution': [],
                              'max_airpollution': [], 
                              'coverage':[],
                              'spray_effect':[],
                              'runtime': 0.0}
            self.save_dir = Setting.save_dir
            os.makedirs(Setting.save_dir, exist_ok=True)
            # self.save_name = f'E{args.env}_D{args.dimension}_FS{args.fire_size}_FN{args.fire_num}_U{args.update_interval}_T{args.horizon}_C{args.communication}_N{args.num_robot}_I{args.image_size}_S{args.suppress_size}_M{args.measure_correct:.2f}_A{args.alpha:.2f}_B{args.beta:.2f}'
            self.save_name = Setting.save_name
            
    def append(self, t, env, observed_env, MI_information, computed_effect, team, coverage, mean_airpollution, max_airpollution, spray_effect):
        self.save_data['time_series'][t] = {
            'state': {id: robot.state for id,robot in team.items()},
            'water_volume': {id: robot.water_volume_now for id,robot in team.items()}
        }
        self.save_data['truth_env'].append(env)
        self.save_data['observed_env'].append(observed_env)
        self.save_data['MI_information'].append(MI_information)
        self.save_data['computed_effect'].append(computed_effect)
        self.save_data['coverage'].append(coverage)
        self.save_data['spray_effect'].append(spray_effect)
        self.save_data['mean_airpollution'].append(mean_airpollution)
        self.save_data['max_airpollution'].append(max_airpollution)

    def append_weight(self, t, info_weights, suppress_weights):
        self.save_data['time_series'][t]['info_weights'] = info_weights
        self.save_data['time_series'][t]['suppress_weights'] = suppress_weights

    def append_confweight(self, t, conf_weights):
        self.save_data['time_series'][t]['conf_weights'] = conf_weights

    def save(self, runtime) -> None:
        self.save_dir = makefile(f'{self.save_dir}/{self.save_name}')
        # self.save_dir = f'{self.save_dir}/{self.save_name}.pkl'
        self.save_data['save_name'] = self.save_name
        self.save_data['runtime'] = runtime
        with open(f"{self.save_dir}", 'wb') as handle:
            pickle.dump(self.save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print()
        print(f"Saved log.json to {self.save_dir}")
