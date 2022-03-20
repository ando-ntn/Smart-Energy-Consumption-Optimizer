# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 23:50:03 2022

CONFIG FILE

@author: Xandrous
"""

import numpy as np
"""
lenhouse = House length(m)
widhouse = House width(m)
hthouse = House height(m)
pitroof = Roof pitch (degree)
numwindows = Number of windows
htwindows = Height of windows(m)
widwindows = Width of windows(m)
"""

r2d = 180/np.math.pi
lenhouse = 30
widhouse = 10
hthouse = 4
pitroof = 40/r2d
numwindows = 6
htwindows = 1
widwindows = 1

# time_sample --> the device takes data every n seconds
time_sample = 60
"house geometry (window area and wall area)"
windowarea = numwindows*htwindows*widwindows
wallarea = 2*lenhouse*hthouse + 2*widhouse*hthouse + 2*(1/np.math.cos(pitroof/2))*widhouse*lenhouse + np.math.tan(pitroof)*widhouse - windowarea

"""
Insulation used and equivalent thermal resistance (req)
Glass wool in the walls, 0.2 m thick

k is in units of J/sec/m/C - convert to J/hr/m/C multiplying by 3600

Glass windows, 0.01 m thick
"""
kwall = 0.038*time_sample
lwall = .2
rwall = lwall/(kwall*wallarea)

kwindow = 0.78*time_sample
lwindow = .01
rwindow = lwindow/(kwindow*windowarea)

req = rwall*rwindow/(rwall + rwindow)

"""Property of cooling system and air with cost of electricity and efficiency
c = cp of air (273 K) = 1005.4 J/kg-K

theater = temperature out of heater/cooler = 3

densair = density of air = 1.2250 kg/m^3

mdot = air flow rate = kg/sec

with 10 min as the time unit

M = total internal air mass

TinIC = initial indoor temperature = 28 deg C

1 kwh = 3.6 x 10^6 J cost = 0.09 USD / kwh = 0.09 / 3.6e6

eff_heater = heater efficiency = 80 %
"""
c = 1005.4
theater= 15
densair = 1.225
mdot = 6
M = (lenhouse*widhouse*hthouse + np.math.tan(pitroof)*widhouse*lenhouse) * densair

TinIC = 28 

cost = 0.09 / 3.6e6

"LEARNING AND RL PARAMETERS"
batch_size = 40
gamma = 0.9
lr= 0.01    
"action_space in this case we just mention the T_set and Fan Speed Mdot combination (8 t_set * 3 mode with 12 force off condition)"
action_space = 25
state_space = 2
epsilon = 0.1
"power preferences"
alpha = 1

"Generate outdoor data"
t_set = 25
t_diff = 1
t_out_avg = 28
t_out_diff = 5
"indicates total sample in specific time frame. 1440/1440 means 1440 samples in 1 day worth of duration"
time_sample_hours = 1440