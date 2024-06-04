import time
import itertools
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import interpolate
from scipy.stats import multivariate_normal
warnings.filterwarnings("ignore")

np.set_printoptions(precision=2)

import json
import copy
import torch
import pandas as pd
pd.set_option('display.max_columns', None)

f2 = open('info.json', 'r')
info_data = json.load(f2)


train_range = 360
test_range = 180
update_time_range = 60


layers = [3, 30, 30, 30, 30, 4]
diffusion_k = info_data["diffusion_k"]
# diffusion_k = 0.02
Iloss_min = 0.0000001
learning_rate = 0.001
learning_rate_LBFGS = 0.1
information_iteration = 10
show_iteration = 5000
save_iteration = 100
I_num = 1000
I_num_sprayer_mul = 0.5
I_num_sprayer_select_mul = 0.01
I_sample_var = 3
regularization_lambda = info_data["regularization_lambda"]
# regularization_lambda = 0.5
sprayer_lambda = info_data["sprayer_lambda"]
# sprayer_lambda = 0.1
training_iteration = 10000
training_iteration_LGBFS = 0
resample_iteration = 100
sprayer_model_mean = [0.0, 0.0]
sprayer_model_variance_num = info_data["sprayer_model_variance_num"]
sprayer_model_variance = [sprayer_model_variance_num, sprayer_model_variance_num]

device_name = "cpu"

lat_range = 150 # latitude grid range (Unit: km)
lon_range = 150 # longitude grid range (Unit: km)
lb_x = 0
lb_y = 0
ub_x = 140
ub_y = 90

test_show_row = 2



class Network(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, 20)
        self.fc5 = torch.nn.Linear(20, 1)

    def forward(self, input_layer):
        hid = torch.tanh(self.fc1(input_layer))
        hid = torch.tanh(self.fc2(hid))
        hid = torch.tanh(self.fc3(hid))
        hid = torch.tanh(self.fc4(hid))
        return self.fc5(hid)


class InformedNN:
    # Initialize the class
    def __init__(self, now_t, info_change = None):
        # Loss function hyperparameters
        network = Network(layers)
        self.network = network.to(device = device_name)
        if info_change == None:
            self.lambda_I = regularization_lambda
            self.lambda_S = sprayer_lambda
            self.diffusion_k = diffusion_k
            self.sprayer_model_variance = sprayer_model_variance
        else:
            self.lambda_I = info_change["regularization_lambda"]
            self.lambda_S = info_change["sprayer_lambda"]
            self.diffusion_k = info_change["diffusion_k"]  
            self.sprayer_model_variance = [info_change["sprayer_model_variance_num"], info_change["sprayer_model_variance_num"]]
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
        data = np.array(data)
        T =  torch.tensor((data[:,0:1] - self.T_mean) / self.T_std, dtype=torch.float)
        X =  torch.tensor((data[:,1:2] - self.X_mean) / self.X_std, dtype=torch.float)
        Y =  torch.tensor((data[:,2:3] - self.Y_mean) / self.Y_std, dtype=torch.float)
        return self.network(torch.hstack([T, X, Y])) * self.D_std + self.D_mean

    def I_generate(self, data_sprayer_train):
        I_num_each_sprayer = int(I_num * I_num_sprayer_mul / len(data_sprayer_train))
        I_S = None
        for i in range(len(data_sprayer_train)):
            sprayer_z = (pd.DataFrame(data_sprayer_train[i]) - [self.T_mean, self.X_mean, self.Y_mean, 0])\
                                        / [self.T_std, self.X_std, self.Y_std, 1]
            self.sprayer_z = sprayer_z
            if sprayer_z.shape[0] < 2:
                continue
            I_new = np.vstack([np.random.random(int(I_num_each_sprayer / I_num_sprayer_select_mul))\
                               * (sprayer_z.time.max() - sprayer_z.time.min())+ sprayer_z.time.min(),\
                                    np.random.normal(0, I_sample_var, int(I_num_each_sprayer / I_num_sprayer_select_mul)),\
                                    np.random.normal(0, I_sample_var, int(I_num_each_sprayer / I_num_sprayer_select_mul))]).T
            I_spray_compute = pd.DataFrame(I_new, columns = ["time","x_I","y_I"])
            I_spray_compute["x_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["x"])(I_spray_compute.time) - I_spray_compute.x_I
            I_spray_compute["y_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["y"])(I_spray_compute.time) - I_spray_compute.y_I
            I_spray_compute["distance"] = I_spray_compute["x_drift"] ** 2 + I_spray_compute["y_drift"] ** 2
            I_new = I_spray_compute.sort_values(by = "distance").head(I_num_each_sprayer).loc[:,["time","x_I","y_I"]]
            I_S = pd.concat([I_S,I_new])
            
        I_noS = pd.DataFrame(np.vstack([np.random.normal(0, I_sample_var, I_num - I_S.shape[0]),\
                                   np.random.normal(0, I_sample_var, I_num - I_S.shape[0]),\
                                   np.random.normal(0, I_sample_var, I_num - I_S.shape[0])]).T, columns = ["time","x_I","y_I"])
        return np.array(pd.concat([I_S,I_noS]))
        
    def sprayer(self, I, data_sprayer_train):
        I = pd.DataFrame(I,columns = ["time","x_I","y_I"])
        I = I.assign(S = pd.Series([0 for j in range(len(I.index))]).values)
        for i in range(len(data_sprayer_train)):
            sprayer_z = (pd.DataFrame(data_sprayer_train[i]) - [self.T_mean, self.X_mean, self.Y_mean, 0])\
                            / [self.T_std, self.X_std, self.Y_std, 1]
            self.sprayer_z = sprayer_z
            if sprayer_z.shape[0] < 2:
                continue
                
            I_spray_compute = I.loc[(I.time < sprayer_z.time.max())\
                                      & (I.time >= sprayer_z.time.min()),:]
            I_spray_compute["x_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["x"])(I_spray_compute.time) - I_spray_compute.x_I
            I_spray_compute["y_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["y"])(I_spray_compute.time) - I_spray_compute.y_I
            I_spray_compute["spray_volume"] = interpolate.interp1d(sprayer_z.time,sprayer_z["spray_volume"])(I_spray_compute.time)
            I_spray_compute["S"] = multivariate_normal.pdf(I_spray_compute.loc[:,["x_drift","y_drift"]],\
                                           mean=np.array(sprayer_model_mean),\
                                           cov=np.diag(np.array(self.sprayer_model_variance)**2)) * I_spray_compute["spray_volume"]
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
            D_0 = (0 - self.D_mean) / self.D_std
            S_m = self.lambda_S * S * ((I_output.detach() - D_0) ** 2)
            I_re = self.diffusion_k * (I_xx + I_yy) - I_t - S_m
            self.I_output = I_output.mean()
            self.S_m = S_m.mean()
            self.Dloss = torch.pow(self.network(torch.hstack([T,X,Y])) - D, 2).mean()
            self.Dloss_set.append(self.Dloss.item())
            self.Iloss = torch.pow(I_re, 2).mean()
            self.Iloss_set.append(self.lambda_I * self.Iloss.item())

            self.loss = self.Dloss + self.lambda_I * self.Iloss
            self.loss_set.append(self.loss.item())
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def test_show(self, train_range, test_range, update_time_range, result_path = None, time_start = None, show_resolution = (60,60),\
                  show_range_z_score = 3, fig_size = 4, vmin = 0, vmax = 40):
        x_range = np.arange(-show_range_z_score, show_range_z_score, 2 * show_range_z_score / show_resolution[0])\
                                        * self.X_std + self.X_mean
        y_range = np.arange(-show_range_z_score, show_range_z_score, 2 * show_range_z_score / show_resolution[1])\
                                        * self.Y_std + self.Y_mean
        data_show = itertools.product(np.arange(self.now_t - train_range, self.now_t + test_range, update_time_range),\
                               x_range, y_range)
        data_show = pd.DataFrame([x for x in data_show])
        data_show.columns = ['time', 'x', 'y'] 
        data_show["pm2d5_after_cal"] = self.forward(np.array(data_show)).detach().numpy()
        subfig_num = int((train_range + test_range) / update_time_range)
        data_show_matrix = [np.zeros(show_resolution) for i in np.arange(-train_range, test_range, update_time_range)]
        imax = data_show.shape[0]
        print("Showing the PINNs' model ...")
        for i,j in data_show.iterrows():
            if i % 1000 == 0:
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d%%" % ('=' * (int(i*100/imax) // 2), int(i*100/imax)))
                sys.stdout.flush()
            t_index = round((j.time - self.now_t + train_range) / update_time_range)
            x_index = round(((j.x - self.X_mean) / self.X_std + show_range_z_score) * show_resolution[0] / 2 / show_range_z_score)
            y_index = round(((j.y - self.Y_mean) / self.Y_std + show_range_z_score) * show_resolution[1] / 2 / show_range_z_score)
            data_show_matrix[t_index][x_index][y_index] = j.pm2d5_after_cal 
        for i,j in self.data_train.iterrows():
            t_index = round((j.time - self.now_t + train_range) / update_time_range)
            if t_index < 0 or t_index >= 60:
                continue
            x_index = round(((j.x - self.X_mean) / self.X_std + show_range_z_score) * show_resolution[0] / 2 / show_range_z_score)
            if x_index < 0 or x_index >= 60:
                continue
            y_index = round(((j.y - self.Y_mean) / self.Y_std + show_range_z_score) * show_resolution[1] / 2 / show_range_z_score)
            if y_index < 0 or y_index >= 60:
                continue
            data_show_matrix[t_index][x_index][y_index] = j.pm2d5_after_cal 
        # data_pre = data_train.loc[data_train.time == 252359,:]
        self.data_show_matrix = data_show_matrix
        subfig_onerow = int(subfig_num / test_show_row)
        # plt.figure(figsize = [fig_size * test_show_row, fig_size * subfig_onerow])
        fig, axes = plt.subplots(test_show_row, subfig_onerow, figsize=(fig_size * subfig_onerow, fig_size * test_show_row))
        fig.subplots_adjust(wspace = 0.4, hspace = 0.2)
        line_color = np.random.random(3)
        line_x = [[]for i in range(len(self.data_sprayer_train))]
        line_y = [[]for i in range(len(self.data_sprayer_train))]
        for i, t_show in zip(range(subfig_num), np.arange(self.now_t - train_range, self.now_t + test_range, update_time_range)):
            i_row = int(i / subfig_onerow)
            i_col = int(i % subfig_onerow)
#             ax = plt.subplot(test_show_row, subfig_onerow, i + 1, figsize=(10, 10))
            if i % subfig_onerow != 0:
                axes[i_row][i_col].yaxis.set_major_locator(plt.NullLocator())
            else:
                axes[i_row][i_col].set_ylabel("y")
            axes[i_row][i_col].set_xlim([0, show_resolution[0]])
            axes[i_row][i_col].set_xticks(np.linspace(0, show_resolution[0], 4))
            axes[i_row][i_col].set_xticklabels(np.around(np.linspace(x_range.min(), x_range.max(), 4),2))
            axes[i_row][i_col].set_ylim([0, show_resolution[1]])
            axes[i_row][i_col].set_yticks(np.linspace(0, show_resolution[1], 4))
            axes[i_row][i_col].set_yticklabels(np.around(np.linspace(y_range.min(), y_range.max(), 4),2))
        #     ax = plt.axes()
        #     ax.xaxis.set_major_locator(plt.NullLocator())
        #     ax.yaxis.set_major_locator(plt.NullLocator())
        #     plt.figure(figsize = [4.2,2.7])
            im = axes[i_row][i_col].imshow(np.matrix(data_show_matrix[i]).T, cmap='turbo',vmin = vmin, vmax = vmax)
#             if i == subfig_num - 1:
#                 axes[i_row][i_col].colorbar()
            if time_start is None:
                time_str = str(t_show - self.now_t) + "min"
            else:
                time_str = str(time_start + pd.Timedelta(pd.offsets.Second(t_show * 60)))[:-3]
            axes[i_row][i_col].set_title(time_str)
            for j in range(len(self.data_sprayer_train)):
                self.t_show = t_show
                self.j = j
                sprayer_point_range = self.data_sprayer_train[j].loc[(self.data_sprayer_train[j].time - t_show).abs() <= update_time_range,:]
                if sprayer_point_range.shape[0] == 0:
                    continue
                sprayer_point = sprayer_point_range.loc[(sprayer_point_range.time - t_show).abs().idxmin()]
#                 sprayer_point = self.data_sprayer_train[j].loc[(self.data_sprayer_train[j].time >= t_show)&
#                                                           (self.data_sprayer_train[j].time < t_show + update_time_range),:]
                x_index = ((sprayer_point.x - self.X_mean) / self.X_std + show_range_z_score) * show_resolution[0] / 2 / show_range_z_score
                line_x[j].append(x_index)
                y_index = ((sprayer_point.y - self.Y_mean) / self.Y_std + show_range_z_score) * show_resolution[1] / 2 / show_range_z_score
                line_y[j].append(y_index)
                axes[i_row][i_col].scatter(x_index,y_index,s = 0.001 * sprayer_point_range.spray_volume.sum(),\
                                           color = line_color * j % 0.5 + 0.5,label = "sprayer " + str(j))
#                 axes[i_row][i_col].plot(line_x[j],line_y[j],color = line_color * j % 0.5 + 0.5)
        fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
#         plt.legend()
        if result_path is not None:
            plt.savefig(result_path + "/" + str(self.now_t) + ".png")
        plt.show()
        # Loss function hyperparameters
        
    def train_show(self):
        plt.ylim([0,1])
        plt.scatter(range(len(self.Dloss_set)), self.Dloss_set,label = "Dloss",s=1)
        plt.scatter(range(len(self.Dloss_set)), self.Iloss_set,label = "Iloss",s=1)
        plt.scatter(range(len(self.Dloss_set)), self.loss_set,label = "loss",s=1)
        plt.legend()
        plt.show()
 
    def train(self, data_LS, data_sprayer_train, show = False):
        self.network.train()
        self.data_sprayer_train = data_sprayer_train
        self.data_train = data_LS
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
        print("Training the PINNs' model...")
        for i in range(training_iteration):
            if i % resample_iteration == 0:
#                 I = np.vstack([np.random.normal(0, I_sample_var, I_num),\
#                                    np.random.normal(0, I_sample_var, I_num),\
#                                    np.random.normal(0, I_sample_var, I_num)]).T
                I = self.I_generate(data_sprayer_train)
                S = self.sprayer(I, data_sprayer_train)
                I = torch.tensor(I, dtype=torch.float, requires_grad=True, device = device_name)
            self.learn(T, X, Y, D, I, S)
            if i % information_iteration == 0:
                if show == True:
#                     sys.stdout.write('\r')
#                     sys.stdout.write("[%-50s] %d%%" % ('=' * (int(i*100/training_iteration) // 2), int(i*100/training_iteration)))
#                     sys.stdout.flush()
                    train_time = time.time() - self.now_time
                    self.now_time = time.time()
                    print("iteration:%d, loss:%f, Iloss:%f, Dloss:%f, S_m:%f, I_output:%f, time:%f"%\
                          (i, self.loss, self.Iloss, self.Dloss, self.S_m,self.I_output, train_time))
#             if i % show_iteration == 0:
#                 self.train_show()
#             if i % save_iteration == 0:
# #                 time_now = time.time()
#                 if self.loss < self.bestloss:
# #                     print("save....")
#                     self.best_model = copy.deepcopy(self.network)
#                     self.bestloss = self.loss
#                 if self.Iloss <= Iloss_min:
#                     self.network = copy.deepcopy(self.best_model)
        if show == True:
            print("\n")
            self.train_show()
        self.network = self.network.to(device = 'cpu')