# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:01:03 2022

Simulation handler

@author: Xandrous
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import config
import time
"import core_algorithm as core"
import functional_API as func
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


t_set = config.t_set
t_diff = config.t_diff
t_out_avg = config.t_out_avg
t_out_diff = config.t_out_diff
"indicates total sample in specific time frame. 1440/1440 means 1440 samples in 1 day worth of duration"
time_sample_hours = config.time_sample_hours

x_t_outdoor = np.linspace(-1/2*np.pi, 3/2*np.pi*time_sample_hours/1440, num = time_sample_hours)
t_outdoor = t_out_avg + t_out_diff*np.sin(x_t_outdoor)
time_outdoor = np.linspace(0,time_sample_hours-1 , num = time_sample_hours)


"init Subsytem function"

start_time = time.time()

print('Welcome to DQN Algorithm for SECO')
mode = input('Config Mode (Deploy/Train): ')


functional = func.functional(config.TinIC)

if mode == "Train" : 
    for i in range (0, time_sample_hours-1) :
        functional.TrainMode(t_outdoor = t_outdoor[i]) 
    "scaler enter the chat"
    scaler_temp = StandardScaler()
    scaler_watt = StandardScaler()
    scaled_temp = scaler_temp.fit_transform(np.reshape(t_outdoor, (-1,1)))
    scaled_watt = scaler_watt.fit_transform(np.reshape(functional.q_heater, (-1,1)))
    scaler_temp_filename = "scaler_temp.save"
    scaler_watt_filename = "scaler_watt.save"
    joblib.dump(scaler_temp, scaler_temp_filename) 
    joblib.dump(scaler_watt, scaler_watt_filename)
    
elif mode == "Deploy" : 
    
#     scaler function is in the init function of functional class
    
    for i in range (0, time_sample_hours) : 
        functional.DeployMode(t_outdoor = t_outdoor[i])
    scaler_temp = StandardScaler()
    scaler_watt = StandardScaler()
    scaled_temp = scaler_temp.fit_transform(np.reshape(functional.t_room, (-1,1)))
    scaled_watt = scaler_watt.fit_transform(np.reshape(functional.q_heater, (-1,1)))
    scaler_temp_filename = "scaler_temp.save"
    scaler_watt_filename = "scaler_watt.save"
    joblib.dump(scaler_temp, scaler_temp_filename) 
    joblib.dump(scaler_watt, scaler_watt_filename) 
print ("Total Reward : ", np.sum(functional.reward_step))

"""action_taken = []
for a in range (len(replay.experiences)) : 
    action_taken.append(replay.experiences[a][2])
    
print(action_taken)"""


print("--- %s seconds ---" % (time.time() - start_time))

"VISUALIZING THE RESULT (T at room vs T Outdoor)"
plt.plot(time_outdoor, t_outdoor, c='red', label = 'Temp_outdoor')
plt.plot(time_outdoor, functional.t_room, label = 'Temp_room')
plt.title('T Room vs T outdoor with On/off set temp', size = 15)
plt.xlabel('Time Frame (per 1 mins)')
plt.ylabel('Temperature in C')
plt.legend()

"visualizing the result (Q Heater and Cost)"
total_energy = np.sum(functional.q_heater)
graph_q = []
graph_q.append(0)
for i in range (1, len(functional.q_heater)+1) : 
    graph_q.append(graph_q[i-1] + functional.q_heater[i-1])
del graph_q[0]

plt.plot(time_outdoor, functional.q_heater, label = 'Instantaneous Power at Time T')
plt.plot(time_outdoor, graph_q, c='red', label = 'Accumulated Energy Over Time')
plt.title('Total Energy in On/off set temp', size = 15)
plt.xlabel('Time Frame (per  mins)')
plt.ylabel('kWH')
plt.legend()
print (total_energy, "kWh")

"Visualizing Reward over time"
graph_reward = []
graph_reward.append(0)
for i in range (1, len(functional.reward_step)+1) : 
    graph_reward.append(graph_reward[i-1] + functional.reward_step[i-1])
del graph_reward[0]

plt.plot(time_outdoor, graph_reward, label = "reward over time")
plt.title('Reward Graph', size = 15)
plt.xlabel('Time Frame (per  mins)')
plt.ylabel('units')
plt.legend()