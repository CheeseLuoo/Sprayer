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
import Sprayer_PINN as SP


f2 = open('info.json', 'r')
info_data = json.load(f2)


train_range = 360
test_range = 60
update_time_range = 60

time_start = pd.Timestamp('2022-09-18 00:00:00')
time_start_2 = pd.Timestamp('2022-12-31 00:00:00')
end_time = int((time_start_2 - time_start).total_seconds()/60)

device_name = "cpu"

time_slice = 1
x_slice = 0.1
y_slice = 0.1





vehicle_n = 22
static_n = 17
refer_n = 12





result_path = 'result'
process_path = 'process'
isExists = os.path.exists(result_path)
if not isExists:
    os.makedirs(result_path)
    os.makedirs(process_path)
else:
    result_path = 'result_'+ time.strftime('%H_%M_%S',time.localtime(time.time()))
    process_path = 'process_'+ time.strftime('%H_%M_%S',time.localtime(time.time()))
    isExists = os.path.exists(result_path)
    if not isExists:
        os.makedirs(result_path)
        os.makedirs(process_path)

print("read vehicle data...")
data_m = []
data_m_ifrefer = [0 for i in range(vehicle_n)]
data_m_test_result = [0 for i in range(vehicle_n)]
data_m_total_test_result = [0 for i in range(vehicle_n)]
i = 0
for root, dirs, files in os.walk("../Guicheng_data/Station"):
    for file in files:
#         print(os.path.join(root, file))
        data_m.append(pd.read_csv(os.path.join(root, file)))
        data_m[i] = data_m[i].iloc[:,[0,1,7,8]].dropna().reset_index(drop = True)
        data_m[i]["pm2d5_after_cal"] = data_m[i].pm2d5
        data_m[i].time = pd.to_datetime(data_m[i].date) - time_start
        data_m[i]["date"] = pd.to_timedelta(list(pd.to_datetime(data_m[i].date) - time_start)).total_seconds() / 60
        data_m[i] = data_m[i].rename(columns={'date': 'time', 'lon': 'x', 'lat': 'y'})
        i += 1

static_n = i

# print("exclude the rainy days...")
# rain_time = ['2022-09-25','2022-09-27','2022-09-28',
#             '2022-10-04','2022-10-05','2022-10-06','2022-10-07','2022-10-08','2022-10-26',
#             '2022-11-12','2022-11-14','2022-11-16','2022-11-17','2022-11-18','2022-11-21','2022-11-22','2022-11-28','2022-11-29']
# rain_num_range = []
# i_pre = start_time_2 - 1440
# for i in rain_time:
#     rain_t = int((pd.Timestamp(i + ' 00:00:00+0800') - time_start).total_seconds() / 60)
#     rain_num_range.append(range(i_pre + 1440 + train_range, rain_t - test_range, update_time_range))
#     i_pre = rain_t

print("read sprayer data...")
sprayer_parameter = pd.read_csv("../Guicheng_data/Volume.csv")
data_sprayer = []
i = 0
for root, dirs, files in os.walk("../Guicheng_data/Vehicle"):
    for file in files:
        sprayer_ID = os.path.join(root, file)[-15:-8]
        print(sprayer_ID)
        sprayer_scale = sprayer_parameter.loc[sprayer_parameter["车牌"] == sprayer_ID,["水箱高度", "水箱吨位"]]
        print(sprayer_scale)
        data_sprayer.append(pd.read_csv(os.path.join(root, file)))
        data_sprayer[i] = data_sprayer[i].iloc[:,[0,2,3,6,7]].dropna().reset_index(drop = True)
        data_sprayer[i].time = pd.to_datetime(data_sprayer[i].date) - time_start
        data_sprayer[i]["date"] = pd.to_timedelta(list(pd.to_datetime(data_sprayer[i].date) - time_start)).total_seconds() / 60
        data_sprayer[i] = data_sprayer[i].rename(columns={'date': 'time',\
                                                          'longitude': 'x', 'latitude': 'y', 'level_diff_process':'spray_volume'})
        data_sprayer[i]["spray_volume"] = - data_sprayer[i]["spray_volume"] * int(sprayer_scale["水箱吨位"]) / int(sprayer_scale["水箱高度"])
        i += 1 
sprayer_n = i
# raise

train_static_num_set = []
train_sprayer_num_set = []
test_static_num_set = []
test_sprayer_num_set = []
# end_time = 8000
# while True:
#     result_path = 'result'
#     process_path = 'process'
#     isExists = os.path.exists(result_path)
#     if not isExists:
#         os.makedirs(result_path)
#         os.makedirs(process_path)
#     else:
#         result_path = 'result_'+ time.strftime('%H_%M_%S',time.localtime(time.time()))
#         process_path = 'process_'+ time.strftime('%H_%M_%S',time.localtime(time.time()))
#         isExists = os.path.exists(result_path)
#         if not isExists:
#             os.makedirs(result_path)
#             os.makedirs(process_path)
#     regularization_lambda_seed = np.random.random()
# #     regularization_lambda = 0.1 * pow(100, regularization_lambda_seed * 2 - 1) # 0.001~10
#     regularization_lambda = 0.02

#     diffusion_k_seed = np.random.random()
#     diffusion_k = 0.01 * pow(10, diffusion_k_seed * 2 - 1) # 0.001~0.1
#     diffusion_k = 0.01
    
#     sprayer_model_variance_num_seed = np.random.random()
#     sprayer_model_variance_num = 0.3 * pow(3, sprayer_model_variance_num_seed * 2 - 1) # 0.1~0.9
#     sprayer_model_variance_num = 0.1
    
#     sprayer_lambda_seed = np.random.random()
#     sprayer_lambda = 0.1 * pow(100, sprayer_lambda_seed * 2 - 1) # 0.001~10
#     sprayer_lambda = 0.005

#     # 创建字典
#     info_dict = {'diffusion_k': diffusion_k,\
#                  'sprayer_model_variance_num': sprayer_model_variance_num,\
#                  'sprayer_lambda': sprayer_lambda,\
#                  'regularization_lambda': regularization_lambda}
#     info_json = json.dumps(info_dict,sort_keys = False, indent = len(info_dict), separators=(',', ': '))
#     f = open(result_path + '\info.json', 'w')
#     f.write(info_json)
#     f.close()
    
for t in range(train_range, end_time - test_range, 60):
    print(str(time_start + pd.Timedelta(pd.offsets.Second(t*60))))
    data_train = None
    train_static_num = 0
    for i in range(static_n):
        data_pre_new_update = data_m[i].loc[(data_m[i].time >= t - train_range)\
                            &(data_m[i].time < t),['time', 'x', 'y', "pm2d5_after_cal"]]
        data_pre_new_update["sensor_index"] = i
        data_train = pd.concat([data_train, data_pre_new_update])
    if data_train is not None:
        train_static_num = data_train.shape[0]
#     train_static_num_set.append(train_static_num)

    data_sprayer_train = []
    train_sprayer_num = 0
    for i in range(sprayer_n):
        data_pre_new_update = data_sprayer[i].loc[(data_sprayer[i].time >= t - train_range)\
                            &(data_sprayer[i].time < t + test_range),['time', 'x', 'y', "spray_volume"]]
        data_pre_new_update.loc[data_pre_new_update['spray_volume'] < 0, 'spray_volume'] = 0
        if data_pre_new_update.shape[0] > 0:
            data_sprayer_train.append(data_pre_new_update)
        train_sprayer_num += data_pre_new_update.shape[0]
#     train_sprayer_num_set.append(train_sprayer_num) 

    data_test = None
    test_static_num = 0
    for i in range(static_n):
        data_pre_new_update = data_m[i].loc[(data_m[i].time >= t)\
                            &(data_m[i].time < t + test_range),['time', 'x', 'y', "pm2d5_after_cal"]]
        data_pre_new_update["sensor_index"] = i
        data_test = pd.concat([data_test, data_pre_new_update])
    if data_test is not None:
        test_static_num = data_test.shape[0]
#     test_static_num_set.append(test_static_num) 

    if train_sprayer_num > 6000 and test_static_num > 0 and train_static_num > 0:
        print("train_static_num %d"%(train_static_num))
        print("train_sprayer_num %d"%(train_sprayer_num))
        print("test_static_num %d"%(test_static_num))
        model = SP.InformedNN(now_t = t)
        model.train(data_train, data_sprayer_train)
        data_test["prediction_result"] = model.forward(data_test).detach().numpy()
#             train_range_show = 60 
#             test_range_show = 60
#             update_time_range_show = 10
#             model.test_show(train_range_show, test_range_show, update_time_range_show, result_path, time_start)
        data_test.to_csv(result_path + "/data_test_" + str(t) + ".csv", index = False)

    