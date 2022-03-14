# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:35:45 2022

@author: User
"""
import numpy as np

class subsystem () :     
    def __init__ (self, theater, mdot, c, M, req, TinIC, t_set, t_diff, time_sample ) : 
        
        self.t_set = t_set
        self.troom = TinIC
        self.theater = theater
        self.mdot = mdot
        self.c = c
        self.M = M
        self.req = req
        self.t_diff = t_diff
        self.t_bool = False
        self.heat_loss = 0
        self.time_sample = time_sample
        
        
    def thermostat_relay (self, troom, t_outdoor,t_set_mode, mdot_mode, force_off) : 
        # works if temperature higher than t set + diff and off when temperature below t set - diff
        # positive delta_t = COLD ROOM --> off cooler, negative delta_t = HOT ROOM --> on cooler
        
        self.troom = troom
        self.t_outdoor = t_outdoor
        self.t_set = t_set_mode
        self.mdot = mdot_mode
        delta_t = self.t_set - self.troom
        print("delta_T = ", np.round(delta_t,2), " room = ", self.troom, " set = ", self.t_set, " outdoor = ", self.t_outdoor)
        
        if delta_t < -self.t_diff and force_off == False :
            self.t_bool = True 
            print('Thermostat True')

        return self.heater()
    
    def thermostat_auto (self, troom, t_outdoor, auto_bool) : 
#       it will either turn on or turn off based on the controller output. action translation of 0 --> turn off, 1 --> turn on
        
        self.troom = troom
        self.t_outdoor = t_outdoor
        self.t_bool = auto_bool 


        return self.heater()
    
    def heater (self) :
        
        if self.t_bool == True : 
            
            t_mix =self.t_set - self.troom 
            q = t_mix * self.mdot*self.time_sample * self.c
            if q > 0 :
                q = 0
            print ("heater on with q = ", np.round(q,2))

        else : 
            q = 0
            print("heater off")
            
        #reset thermostat
        
        self.t_bool = False   
        return self.house(q)
        
    def house (self, q) : 
        self.heat_loss = (self.troom - self.t_outdoor) / (self.req)
        q_in = q - self.heat_loss
       
        delta_temp = q_in/(self.M*self.c)
        t_final = (self.troom + delta_temp)
        
        
        
        print("heat loss : ", self.heat_loss)
        print("delta_temp = ", delta_temp)
        print("----------------------------")
        return t_final,q