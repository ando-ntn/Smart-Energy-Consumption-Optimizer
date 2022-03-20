# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:22:01 2021
Project SECO
@author: Xandrous
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
from collections import deque


"""
replay experience for memory sampling
"""
class DqnReplayBuffer:
    def __init__(self) : 
        self.experiences = deque(maxlen = 1000000)
        #print("replay init done")
        
    
    def record (self, state, next_state, action, reward, done) : 
        self.experiences.append((state, next_state, action, reward, done))
        #print("replay record done")
    
    def sample_batch (self, batch_size) :
        
        sampled_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for record in sampled_batch:
            state_batch.append(record[0])
            next_state_batch.append(record[1])
            action_batch.append(record[2])
            reward_batch.append(record[3])
            done_batch.append(record[4])
        #print ("replay sample batch done")
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)
        
    
    

"""
Deep Q Network as the brain for the system
"""
class DqnAgent : 
    def __init__ (self, state_space, action_space, gamma, lr, epsilon, alpha ) : 
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.model_location = './model'
        self.checkpoint_location = './checkpoints'
        self.mode = "train"
        
        self.q_net = self.build_dqn_model()
        self.target_q_net = self.build_dqn_model()
        
        "for reward policy, alpha = power usage preferences"
        self.power_usage = []
        self.alpha = alpha
        
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              net=self.q_net)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_location, max_to_keep=10)
        if self.mode == 'train':
            self.load_checkpoint()
            self.update_target_network()
        if self.mode == 'test':
            self.load_model()
            
        #print ("DQN init done")
        
    def build_dqn_model(self) : 
        q_net = keras.Sequential()
        q_net.add(keras.layers.Dense(10, input_dim = self.state_space, activation = 'relu'))
        q_net.add(keras.layers.Dense(20, activation = 'relu'))
        q_net.add(keras.layers.Dense(self.action_space, activation = 'softmax'))
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = self.lr), loss ='mse')
        #print ("DQN build model done")
        return q_net
    
    def save_model(self, reward):
        """
        Saves model to file system
        :return: None
        """
        tf.keras.models.save_model(self.q_net, self.model_location)
        file_reward = open('reward.txt', 'w')
        file_reward.write(np.str(reward))
        file_reward.close()
        print ("DQN save model done")

    def load_model(self):
        """
        Loads previously saved model
        :return: None
        """
        self.q_net = tf.keras.models.load_model(self.model_location)
        print ("DQN load model done")
        
    def save_checkpoint(self):
        """
        Saves training checkpoint
        :return: None
        """
        self.checkpoint_manager.save()
        print ("DQN save ckpt done")

    def load_checkpoint(self):
        """
        Loads training checkpoint into the underlying model
        :return: None
        """
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        print ("DQN load ckpt done")
    
    def update_target_network(self):
        """
        Updates the target Q network with the parameters
        from the currently trained Q network.
        :return: None
        """
        print('DQN update network done')
        self.target_q_net.set_weights(self.q_net.get_weights())
        
    def train(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, batch_size):
        
        """
        Train the model on a batch
        :return: loss history
        
        input state batch into q_net to get current_q 
        target_q values will be copied from current_q
        input next state batch and we will get the next_q
        we will take max values of next_q (our policy)
        """
        action_batch = action_batch.astype('int16')
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis = 1)    
        
        """
        in here we will do assigning values to target_q based on the iteration
        and each of previous action taken. 
        
        target_q[current_batch][action_batch[current_batch]] 
        means that the Q values in specific action taken previously 
        target_q[0][1] means q value when taking action 1 in batch 0
        
        done_batch consist of values either 1 or 0
        
        if 1, we will set target_q on specific action equal to the reward
        
        if 0, we will add reward and discounted value from max next_q 
        (normal Q values calculation)
        """
        for current_batch in range(batch_size) : 
            "to check whether current batch already done or not"
            target_q[current_batch][action_batch[current_batch]] = reward_batch[current_batch] + self.gamma * max_next_q[current_batch]*done_batch[current_batch]
        
        "training the model and save it in history"
        history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        loss = history.history['loss']
        
        
        
        
        
        
        
        print ("DQN Train done")
        return loss
    
    def random_policy(self, state):
        """
        Outputs a random action
        :param state: current state
        :return: action
        """
        print ("DQN random policy done")
        return np.random.randint(0, self.action_space)

    def collect_policy(self, state):
        """
        The policy for collecting data points which can contain 
        some randomness to encourage exploration.
        :return: action
        """
        if np.random.random() < self.epsilon:
            return self.random_policy(state=state)
        print ("DQN collect policy done with epsilon : ", self.epsilon)
        return self.policy(state=state)
    
    def policy(self, state):
        """
        Outputs a action based on model
        :param state: current state
        :return: action
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        optimal_action = np.argmax(action_q.numpy()[0], axis=0)
        return np.int(optimal_action)
    
    def reward_policy(self, user_feedback, current_power) : 
        """
        

        Parameters
        ----------
        feedback_table : user's input 
        if "C" Cold -> -1
        if "N" Normal -> +1
        if "H" Hot -> -1.5
        
        need to define more as in most model RL, rewards are made out of 
        cost function and constraint. in HVAC model-free RL, cost -> Energy Consumption and 
        Constraint -> Comfort Level (temperature and humidty range) need to be combined
        to create reward function as [r = rT + lambda * rp] 
        
        where : rT = cost, lambda = controls tradeoff between cost and constraints
        rp = constraint
        
        Source : https://doi.org/10.1145/3360322.3360861
        
        Returns
        -------
        Reward value

        """
        reward_env = np.int(user_feedback)
        self.power_usage.append(current_power)
        avg_power = np.mean(self.power_usage)
        print ("avg power : %0.02f, current power : %0.02f"% (avg_power, current_power) ) 
        "10 daan 5 itu faktor normalisasi nya"
        reward = self.alpha * (avg_power - current_power) + (-np.abs(reward_env))


        return np.float32(reward)
        
   
