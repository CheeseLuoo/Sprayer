import random

import numpy as np
import torch
import os
import pickle

def seed_everything(Setting):
    random.seed(Setting.seed)
    rng = np.random.RandomState(Setting.seed)
    torch.manual_seed(Setting.seed)
    print(f"Set random seed to {Setting.seed} in random, numpy, and torch.")
    return rng

def print_metrics(logger,select_time):
    print(f'coverage: {np.mean(logger.save_data["coverage"]): .3f}')
    print(f'mean_airpollution: {(logger.save_data["mean_airpollution"][select_time-1]): .3f}')
    print(f'max_airpollution: {logger.save_data["max_airpollution"][select_time-1]: .3f}')

def makefile(dir_name):
    if os.path.exists(f'{dir_name}.pkl'):
        i = 1
        while True:
            new_name = dir_name + "_" + str(i)
            if not os.path.exists(f'{new_name}.pkl'):
                dir_name = new_name
                break
            i += 1
    print("savedir:", dir_name)
    return f'{dir_name}.pkl'

def makedir(dir_name):
    if os.path.exists(f'{dir_name}'):
        i = 1
        while True:
            new_name = dir_name + "_" + str(i)
            if not os.path.exists(f'{new_name}'):
                dir_name = new_name
                break
            i += 1
    print("savedir:", dir_name)
    return f'{dir_name}'

def readpkl(pkldir):
    with open(pkldir, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data