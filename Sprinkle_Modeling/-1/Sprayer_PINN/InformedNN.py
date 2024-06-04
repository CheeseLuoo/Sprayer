import time
import itertools
import os
import sys
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import interpolate
from scipy.stats import multivariate_normal
warnings.filterwarnings("ignore")

np.set_printoptions(precision=2)

import json


f2 = open('info.json', 'r')
info_data = json.load(f2)


train_range = 360
test_range = 180
update_time_range = 60

time_start = pd.Timestamp('2022-04-01 00:00:00+08:00')
time_start_2 = pd.Timestamp('2022-09-23 00:00:00+08:00')
start_time_2 = int((time_start_2 - time_start).total_seconds()/60)

layers = [3, 30, 30, 30, 30, 4]
diffusion_k = info_data["diffusion_k"]
diffusion_k = 0.01
Iloss_min = 0.0000001
learning_rate = 0.001
learning_rate_LBFGS = 0.1
information_iteration = 10
show_iteration = 1000
save_iteration = 100
I_num = 10000
I_sample_var = 3
regularization_lambda = info_data["regularization_lambda"]
regularization_lambda = 0.1
sprayer_lambda = 300
training_iteration = 20000
training_iteration_LGBFS = 0
resample_iteration = 100
sprayer_model_mean = [0.0, 0.0]
sprayer_model_variance = [0.5, 0.5]

device_name = "cpu"

time_slice = 1
x_slice = 0.1
y_slice = 0.1
lat_range = 150 # latitude grid range (Unit: km)
lon_range = 150 # longitude grid range (Unit: km)
lb_x = 0
lb_y = 0
ub_x = 140
ub_y = 90

refer_interpolate_time = 5
refer_interpolate_time_interval = 1


vehicle_n = 22
static_n = 17
refer_n = 12
sprayer_n = 1

start_t = 200000
end_t = 400000

class InformedNN:
    # Initialize the class
    def __init__(self, now_t):
        # Loss function hyperparameters
        network = Network(layers)
        self.network = network.to(device = device_name)
        self.lambda_I = regularization_lambda
        self.lambda_S = sprayer_lambda
        self.now_t = now_t
        self.loss = -1
        self.Dloss = -1
        self.Iloss = -1
        self.Dloss_set = []
        self.Iloss_set = []
        self.loss_set = []
        self.bestloss = 10000000
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)
        self.optimizer_LBFGS = torch.optim.LBFGS(self.network.parameters(), lr = learning_rate_LBFGS)
        self.now_time = time.time()
        
    def forward(self, data):
        T =  torch.tensor((data[:,0:1] - self.T_mean) / self.T_std, dtype=torch.float)
        X =  torch.tensor((data[:,1:2] - self.X_mean) / self.X_std, dtype=torch.float)
        Y =  torch.tensor((data[:,2:3] - self.Y_mean) / self.Y_std, dtype=torch.float)
        return self.network(torch.hstack([T, X, Y])) * self.D_std + self.D_mean

    def sprayer(self, I, data_sprayer_train):
        for i in range(len(data_sprayer_train)):
            sprayer_z = (pd.DataFrame(data_sprayer_train[i]) - [self.T_mean, self.X_mean, self.Y_mean, 0])\
                            / [self.T_std, self.X_std, self.Y_std, 1]
            I = pd.DataFrame(I,columns = ["time","x_I","y_I"])
            I = I.assign(S = pd.Series([0 for j in range(len(I.index))]).values)
            I_spray_compute = I.loc[(I.time < sprayer_z.time.max())\
                                      & (I.time >= sprayer_z.time.min()),:]
            I_spray_compute["x_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["x"])(I_spray_compute.time) - I_spray_compute.x_I
            I_spray_compute["y_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["y"])(I_spray_compute.time) - I_spray_compute.y_I
            I_spray_compute["spray_volume"] = interpolate.interp1d(sprayer_z.time,sprayer_z["spray_volume"])(I_spray_compute.time)
            I_spray_compute["S"] = multivariate_normal.pdf(I_spray_compute.loc[:,["x_drift","y_drift"]],\
                                                    mean=np.array(sprayer_model_mean),\
                                                    cov=np.diag(np.array(sprayer_model_variance)**2)) * I_spray_compute["spray_volume"]
#             print(I_spray_compute)
            I.loc[I_spray_compute.index,"S"] = I.loc[I_spray_compute.index,"S"] + I_spray_compute["S"]
        S = torch.tensor(I["S"], dtype=torch.float, requires_grad=True, device = device_name)
#         print(S)
        return S
    
    def learn(self, T, X, Y, D, I, S, opt = "SGD"):
#         I = torch.tensor(I, dtype=torch.float, requires_grad=True)
#         D_raw = torch.tensor(D_raw, dtype=torch.float, requires_grad=True)
#         E = torch.tensor(E, dtype=torch.float, requires_grad=True)
#         D_refer = torch.tensor(D_refer, dtype=torch.float, requires_grad=True)
#         C = torch.tensor(C, dtype=torch.float, requires_grad=True)
        if opt == "SGD":
            I_output = self.network(I)

            I_x = torch.autograd.grad(I_output, I, torch.ones(I_output.shape),\
                                      retain_graph=True, create_graph=True)[0][:,1]
            I_xx = torch.autograd.grad(I_x, I, torch.ones(I_x.shape),\
                                      retain_graph=True, create_graph=True)[0][:,1]
            I_y = torch.autograd.grad(I_output, I, torch.ones(I_output.shape),\
                                      retain_graph=True, create_graph=True)[0][:,2]
            I_yy = torch.autograd.grad(I_y, I, torch.ones(I_y.shape),\
                                      retain_graph=True, create_graph=True)[0][:,2]
            I_t = torch.autograd.grad(I_output, I, torch.ones(I_output.shape),\
                                      retain_graph=True, create_graph=True)[0][:,0]
            I_re = diffusion_k * (I_xx + I_yy) - I_t - self.lambda_S * S
            self.S_m = (self.lambda_S * S).mean()
            self.Dloss = torch.pow(self.network(torch.hstack([T,X,Y])) - D, 2).mean()
            self.Dloss_set.append(self.Dloss.item())
            self.Iloss = torch.pow(I_re, 2).mean()
            self.Iloss_set.append(self.lambda_I * self.Iloss.item())

            self.loss = self.Dloss + self.lambda_I * self.Iloss
            self.loss_set.append(self.loss.item())
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def test_show(self, train_range, test_range, update_time_range, show_resolution = (60,60),\
                  show_range_z_score = 3, fig_size = 30, vmin = 0, vmax = 40):
        data_result = itertools.product(np.arange(self.now_t - train_range, self.now_t + test_range, update_time_range),\
                               np.arange(-show_range_z_score, show_range_z_score, 2 * show_range_z_score / show_resolution[0])\
                                        * self.X_std + self.X_mean,\
                                np.arange(-show_range_z_score, show_range_z_score, 2 * show_range_z_score / show_resolution[1])\
                                        * self.Y_std + self.Y_mean)
        data_result = pd.DataFrame([x for x in data_result])
        data_result.columns = ['time', 'x', 'y'] 
        data_result["pm2d5_after_cal"] = self.forward(np.array(data_result)).detach().numpy()
        data_show = data_result
        subfig_num = (train_range + test_range) / update_time_range
        data_show_matrix = [np.zeros(show_resolution) for i in np.arange(-train_range, test_range, update_time_range)]
        for i,j in data_show.iterrows():
            if i % 10000 == 0:
                print(i)
            t_index = int((j.time - self.now_t + train_range) / update_time_range)
            x_index = int((j.x - self.X_mean) / self.X_std + show_range_z_score) * show_resolution[0] / 2 / show_range_z_score
            y_index = int((j.y - self.Y_mean) / self.Y_std + show_range_z_score) * show_resolution[1] / 2 / show_range_z_score 
            data_show_1[t_index][x_index][y_index] = j.pm2d5_after_cal 
#         for i,j in data_train.iterrows():
#             data_show_1[int(j.time - t + train_range)][int(j.x)][int(j.y)] = j.pm2d5_after_cal 
        # data_pre = data_train.loc[data_train.time == 252359,:]
        plt.figure(figsize = [fig_size, fig_size * subfig_num])
        ax = plt.axes()
        ax.yaxis.set_major_locator(plt.NullLocator())
        for i, t_show in zip(range(subfig_num), np.arange(self.now_t - train_range, self.now_t + test_range, update_time_range)):
            ax = plt.subplot(1, subfig_num, i + 1)
            if i != 0:
                ax.yaxis.set_major_locator(plt.NullLocator())
            else:
                plt.ylabel("y")
        #     ax = plt.axes()
        #     ax.xaxis.set_major_locator(plt.NullLocator())
        #     ax.yaxis.set_major_locator(plt.NullLocator())
        #     plt.figure(figsize = [4.2,2.7])
            plt.imshow(data_show_1[i], cmap='turbo',vmin = vmin,vmax = vmax)
            if i == subfig_num - 1:
                plt.colorbar()
            plt.title(str(t_show) + "min")
            for j in range(len(self.data_sprayer_train)):
                sprayer_point = data_sprayer_train[j].loc[(data_sprayer_train[j].time >= t_show)&
                                                          (data_sprayer_train[j].time < t_show + update_time_range),:]
                plt.scatter(sprayer_point.x,sprayer_point.y,s = 100,color = "red",label = "sprayer" + str(j))
        # plt.legend()
        plt.show()
#         plt.savefig(result_path + "/" + str(I_num) + " " + str(sprayer_lambda) + " " + str(training_iteration) + ".png")
        # Loss function hyperparameters
 
    def train(self, data_LS, data_sprayer_train):
        self.network.train()
        self.data_sprayer_train = data_sprayer_train
        data_train = np.matrix(data_LS)
        self.T_mean = data_train[:,0].mean()
        self.T_std = data_train[:,0].std() 
        T =  (data_train[:,0] - self.T_mean) / self.T_std
        
        self.X_mean = data_train[:,1].mean()
        self.X_std = data_train[:,1].std() 
        X =  (data_train[:,1] - self.X_mean) / self.X_std
        
        self.Y_mean = data_train[:,2].mean()
        self.Y_std = data_train[:,2].std() 
        Y =  (data_train[:,2] - self.Y_mean) / self.Y_std
        
        self.D_mean = data_train[:,3].mean()
        self.D_std = data_train[:,3].std() 
        D =  (data_train[:,3] - self.D_mean) / self.D_std
        
        T = torch.tensor(T, dtype=torch.float, requires_grad=True, device = device_name)
        X = torch.tensor(X, dtype=torch.float, requires_grad=True, device = device_name)
        Y = torch.tensor(Y, dtype=torch.float, requires_grad=True, device = device_name)
        D = torch.tensor(D, dtype=torch.float, requires_grad=True, device = device_name)

        for i in range(training_iteration):
            if i % resample_iteration == 0:
                I = np.vstack([np.random.normal(0, I_sample_var, I_num),\
                                   np.random.normal(0, I_sample_var, I_num),\
                                   np.random.normal(0, I_sample_var, I_num)]).T
                S = self.sprayer(I, data_sprayer_train)
                I = torch.tensor(I, dtype=torch.float, requires_grad=True, device = device_name)
            self.learn(T, X, Y, D, I, S)
            if i % information_iteration == 0:
                train_time = time.time() - self.now_time
                self.now_time = time.time()
                print("iteration:%d, loss:%f, Iloss:%f, Dloss:%f, S_m:%f, time:%f"%\
                      (i, self.loss, self.Iloss, self.Dloss, self.S_m, train_time))
            if i % show_iteration == 0:
                self.train_show()
            if i % save_iteration == 0:
#                 time_now = time.time()
                if self.loss < self.bestloss:
#                     print("save....")
                    self.best_model = copy.deepcopy(self.network)
                    self.bestloss = self.loss
                if self.Iloss <= Iloss_min:
                    self.network = copy.deepcopy(self.best_model)
        
        self.network = network.to(device = 'cpu')