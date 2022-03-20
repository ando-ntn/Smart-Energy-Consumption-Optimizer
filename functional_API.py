# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:41:26 2022

@author: Xandrous
"""

import core_algorithm as core 
import subsystem as subsystem
import config
import numpy as np
import pandas as pd

class functional() :
    def __init__(self, TinIC) : 
        self.sub = subsystem.subsystem(config.theater, config.mdot, config.c, config.M, config.req, config.TinIC, config.t_set, config.t_diff, config.time_sample)
        self.agent = core.DqnAgent(action_space = config.action_space , state_space = config.state_space, 
                 gamma = config.gamma, lr = config.lr, epsilon = config.epsilon, alpha = config.alpha)
        self.replay = core.DqnReplayBuffer()
        try :
            self.agent.load_model()
        except : 
            print ('no self.agent model found')
        self.state = None
        t_room_initial = TinIC
        self.t_room = []
        self.q_heater = []
        self.cost_heater = []
        self.done_batch =[]
        self.counter = np.int(0)

        self.reward_index = []
        self.reward_step =[]

        self.t_room.append(t_room_initial)
        self.q_heater.append(0)
        self.reward_index.append(0)
        self.done_batch.append(0)
    
    def TrainMode (self, total_data, save_model, load_model, show_loss) : 
#         "for training the model after passing certain number of episode"
        if load_model == True : 
            self.agent.load_model()
            
        reward_total = []
        experience = []
        
#       total_data --> Temperature(state), power cons(state), reward, done_batch (no action because periodical learning)

        for i in range (0,len(total_data)-1) : 
            state = total_data[i,:2]
            next_state = total_data[i+1,:2]
    

#           reward from user's feedback (either 1, 0 or -1)
#           action from action taken previously ()
            
            reward = self.agent.reward_policy(total_data[i,2])

            action = self.agent.collect_policy(next_state)
            
            
            done = total_data[i,-1:].astype(int)
            self.replay.record(state, next_state, np.int(action), reward, done)
            print ("eps : %s , Reward : %0.02f" %(i, reward))
            reward_total.append(reward)
            experience.append((self.replay.experiences[-1][0][0], self.replay.experiences[-1][0][1],self.replay.experiences[-1][1][0], self.replay.experiences[-1][1][1]
                                  ,self.replay.experiences[-1][2], self.replay.experiences[-1][3][0]))      
                    
            
            
#             "to make sure training run on batch or else it won't train"
            if len(self.replay.experiences) > config.batch_size : 
                state_batch, next_state_batch, reward_batch, action_batch, done_batch = self.replay.sample_batch(config.batch_size)
                loss, target_q = self.agent.train(state_batch = state_batch, next_state_batch = next_state_batch
                           , action_batch = action_batch, reward_batch = reward_batch
                           , done_batch = done_batch, batch_size = config.batch_size)
                print("loss : ", np.round(loss,5))
#             "save checkpoint every 50 iteration and update target_q_network"
            if i % 50 == 0 : 
                self.agent.save_checkpoint()
                self.agent.update_target_network()
                "decay rate"
                self.agent.epsilon *= 0.8
                
                
        reward = np.sum(reward_total)       
        last_reward_file = open('reward.txt', 'r')
        last_reward = np.float(last_reward_file.read())
        last_reward_file.close()
        pd.DataFrame(np.array(experience), columns = ['Current_state_heat', 'Current_state_watt', 'Next_State_heat', 'Next_State_watt', 'Action', 'Reward']).to_csv("Train_Memory.csv")
        
        if save_model == True and reward > last_reward : 
            self.agent.save_model(reward)
            
    
    def DeployMode (self, t_outdoor) : 
        experience = []        
#             """setelah dia ngelakuin current action, baru dapet reward.
#             reward yang dikirim ketika T+1 ini akan jadi reward(T)"""
                
#             "State (T+1)"   
        t_room = self.t_room[-1]
        q_heat = self.q_heater[-1]
        total_state = np.stack((t_room, q_heat))
        state = np.array(total_state)
        reward = self.agent.reward_policy(self.reward_index[-1], self.q_heater[-1])
        self.reward_step.append(reward)
        action = self.agent.collect_policy(state)
        print(self.counter)
        done = 0
        
        if self.counter != 0 :
#             self.state is the old state and state is the new_state
            self.replay.record(self.state, state, np.int(action), reward, done)
                
            print("state now : ", state)
            print("next action : ", action)
            print("reward : ", reward)
            print('--------------------------------------------')
                
            "nanti disini musti ditambah untuk save data tadi ke dalam local storage"
            "untuk nanti kita bisa save ke database periodically at the end of the day for training"
                
            "ini untuk save  data nya taruh disini dlu. save setiap 50 step"
            experience.append((self.replay.experiences[-1][0][0], self.replay.experiences[-1][0][1],self.replay.experiences[-1][1][0], self.replay.experiences[-1][1][1]
                                  ,self.replay.experiences[-1][2], self.replay.experiences[-1][3]))   
        
            

#             "to make sure training run on batch or else it won't train"
            if len(self.replay.experiences) > config.batch_size : 
                state_batch, next_state_batch, reward_batch, action_batch, done_batch = self.replay.sample_batch(config.batch_size)
                loss = self.agent.train(state_batch = state_batch, next_state_batch = next_state_batch
                           , action_batch = action_batch, reward_batch = reward_batch
                           , done_batch = done_batch, batch_size = config.batch_size)
                print("loss : ", np.round(loss,5))
#             "save checkpoint every 50 iteration and update target_q_network"

            if self.counter % 100 == 0 : 
                self.agent.save_checkpoint()
                self.agent.save_model(np.sum(self.reward_step))
                self.agent.update_target_network()
                "decay rate"
                if self.agent.epsilon > 0.05 : 
                    self.agent.epsilon *= 0.99    
                pd.DataFrame(np.array(experience), columns = ['Current_state_temp', 'Current_state_watt', 'Next_State_heat', 'Next_State_watt', 'Action', 'Reward']).to_csv("Deploy_Memory.csv")
                
#           Action Translation 
#           t_set = 18,19,20,21,22,23,24,25
#           mdot = mode 1 (3 kg/s) ,mode 2 (6kg/s / normal),mode 3 (9kg/s)
#             if (action == 1) : 
#                 auto_bool = True
#             elif (action == 0) : 
#                 auto_bool = False
#           for resetting force off bool
            t_set_mode = 0
            mdot_mode = 0
            force_off = False
            if (action == 0) : 
                force_off = True
            elif (action == 1) : 
                t_set_mode = 18
                mdot_mode = 3
            elif (action == 2) : 
                t_set_mode = 18
                mdot_mode = 6
            elif (action == 3) : 
                t_set_mode = 18
                mdot_mode = 9
            elif (action == 4) : 
                t_set_mode = 19
                mdot_mode = 3                
            elif (action == 5) : 
                t_set_mode = 19
                mdot_mode = 6
            elif (action == 6) : 
                t_set_mode = 19
                mdot_mode = 9
            elif (action == 7) : 
                t_set_mode = 20
                mdot_mode = 3                
            elif (action == 8) : 
                t_set_mode = 20
                mdot_mode = 6
            elif (action == 9) : 
                t_set_mode = 20
                mdot_mode = 9
            elif (action == 10) : 
                t_set_mode = 21
                mdot_mode = 3                
            elif (action == 11) : 
                t_set_mode = 21
                mdot_mode = 6
            elif (action == 12) : 
                t_set_mode = 21
                mdot_mode = 9                
            elif (action == 13) : 
                t_set_mode = 22
                mdot_mode = 3                
            elif (action == 14) : 
                t_set_mode = 22
                mdot_mode = 6
            elif (action == 15) : 
                t_set_mode = 22
                mdot_mode = 9                
            elif (action == 16) : 
                t_set_mode = 23
                mdot_mode = 3                
            elif (action == 17) : 
                t_set_mode = 23
                mdot_mode = 6
            elif (action == 18) : 
                t_set_mode = 23
                mdot_mode = 9
            elif (action == 19) : 
                t_set_mode = 24
                mdot_mode = 3                
            elif (action == 20) : 
                t_set_mode = 24
                mdot_mode = 6
            elif (action == 21) : 
                t_set_mode = 24
                mdot_mode = 9                
            elif (action == 22) : 
                t_set_mode = 25
                mdot_mode = 3                
            elif (action == 23) : 
                t_set_mode = 25
                mdot_mode = 6
            elif (action == 24) : 
                t_set_mode = 25
                mdot_mode = 9                
                
                
#             t_room_final, q_heat = simulation.thermostat_auto(t_room, t_outdoor, auto_bool)
            t_room_final, q_heat = self.sub.thermostat_relay(t_room, t_outdoor, t_set_mode, mdot_mode, force_off)
    
            if t_room_final <= 26 and t_room_final >= 24 : 
                self.reward_index.append(0)
            elif t_room_final > 24 : 
                self.reward_index.append(1) 
            elif t_room_final < 22 : 
                self.reward_index.append(-1)

            self.t_room.append(t_room_final)
            self.q_heater.append(-q_heat/3.6e6)
            self.cost_heater.append(-q_heat*config.cost)
            
#         self.done_batch[self.counter] = 1  
        self.counter += 1    
        self.state = state