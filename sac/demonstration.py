'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''
import os
import sys


sys.path.append('../')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from simulation import Mujoco_prototype

import numpy as np
import time


from demonstration_dataset import ReplayBuffer
from normalised_action import NormalizedActions
from sac_trainer import SAC_Trainer
from Randomizer import body_swap as bs
from Randomizer.mujoco_randomizer import Randomizer

import argparse

dir_ = os.path.dirname(os.getcwd())

arm_ = "jaco2.xml"
visualize = False
env = Mujoco_prototype(dir_,arm_, visualize)


action_dim = 7
action_range = 1

# hyper-parameters for RL training
max_episodes  = 5000000
max_steps = 5

replay_buffer_size = 40
replay_buffer = ReplayBuffer(replay_buffer_size)

rewards     = []


average_rewards = 0

#Action range for each action
ratio_xy = 0.1
ratio_orient = 0.55
ratio_z = 0.15

ratio_ = np.array([ratio_xy, ratio_xy, ratio_z, ratio_orient, ratio_orient, ratio_orient ])
ratio_residual_force = 4

#initialize randomizer
randomizer = Randomizer(env.interface)

#body randomizer
#BodyID
body_cube = randomizer.body(2)
body_cylinder = randomizer.body(3)
light = randomizer.light()

do_we_break = False

eps = 0
average_time = 0

while True:
    
    #randomize the position and orientation every step 
    env.restart()

    

    #randomize position
    geom_body_ID, target_pos, target_orient, size = bs.body_swap(body_cube, body_cylinder)

    _, height, _ = env.get_limit_target_pos(geom_body_ID)


    light._rand_textures()
    light._rand_lights()

    episode_reward = 0


    #I don't want to be too close by the target
    #target_estimated_pos = (target_pos + np.array([0 , 0 , 0.1])).tolist()
    target_estimated_pos = (target_pos + np.array([0.05 , 0.05, 0]*(np.random.rand(3)-0.5)+np.array([0 , 0 , 0.3]))).tolist()
    #target_estimated_orientation = list(target_orient)
    target_estimated_orientation = np.array([0, 0, 0],np.float64)
    initial_gripper_force = np.array([5, 5, 5],np.float64)

    #I initialize and resize the first action

    #estimate how much to shift
    shifted_xyz = np.array(target_estimated_pos - env.get_hand_pos(),np.float64)
    scripted_action = np.array([shifted_xyz, target_estimated_orientation, initial_gripper_force], np.float64)
    scripted_action = np.resize(scripted_action,(9))
    

    _, _, _, _ = env.step_sim(scripted_action, -1,  geom_body_ID)

    action = np.zeros(9, dtype=np.float64)

    start = time.time()

    state_image, state_hand = env.get_state()

    shift_orientation = (target_orient[0]/3, target_orient[1]/3, target_orient[2]/3)

    random_position_shift = 2*np.array([0.1, 0.1])*(np.random.rand(2)-0.5)
    #FIRST ACTION
   
    action[:2] = target_pos[:2] + random_position_shift
    
    action[2] = height + 0.1*np.random.rand(1) + 0.1
    action[:3] = action[:3] - env.get_hand_pos()
    action[3:6] = shift_orientation
    action[6:] = [10,10,10]


    next_state_image, next_state_hand, reward, done = env.step_sim(action, 0, geom_body_ID)
    flag = replay_buffer.push(state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
    episode_reward = episode_reward + reward
    

    state_image = next_state_image
    state_hand = next_state_hand

    if (flag):
        break

    #SECOND ACTION
    
    action[:2] = target_pos[:2] + random_position_shift/2
    
    action[2] = env.get_hand_pos()[2]
    action[:3] = action[:3] - env.get_hand_pos()
    action[3:6] = shift_orientation
    action[6:] = [10,10,10]
    next_state_image, next_state_hand, reward, done = env.step_sim(action, 1, geom_body_ID)
    flag = replay_buffer.push(state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
    episode_reward = episode_reward + reward
    

    state_image = next_state_image
    state_hand = next_state_hand

    if (flag):
        break

    #THIRD ACTION
    
    action[:2] = target_pos[:2] 
    
    action[2] = height + 0.06
    action[:3] = action[:3] - env.get_hand_pos()
    action[3:6] = shift_orientation
    action[6:] = [10,10,10]
    next_state_image, next_state_hand, reward, done = env.step_sim(action, 2, geom_body_ID)
    flag = replay_buffer.push(state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
    episode_reward = episode_reward + reward
    

    state_image = next_state_image
    state_hand = next_state_hand

    if(flag):
        break

    #FOURTH ACTION
    action[:6] = [0,0,0,0,0,0]
    action[6:] = [0.2,0.2,0.2]
    next_state_image, next_state_hand, reward, done = env.step_sim(action, 3, geom_body_ID)
    flag = replay_buffer.push(state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
    episode_reward = episode_reward + reward
    

    state_image = next_state_image
    state_hand = next_state_hand

    if(flag):
        break

    #FIFTH ACTION
    
    action[:6] = [0, 0, 0.3*np.random.rand(1), 0, 0, 0]
    action[6:] = [0.2, 0.2, 0.2]
    next_state_image, next_state_hand, reward, done = env.step_sim(action, 4, geom_body_ID)
    flag = replay_buffer.push(state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
    episode_reward = episode_reward + reward
    

    state_image = next_state_image
    state_hand = next_state_hand  
     
    end = time.time()
    average_time = average_time + end - start
    average_rewards = average_rewards + episode_reward

    if(flag):
        break



    if eps % 5 == 0 and eps>0: # plot and model saving interval
        average_rewards = 0
        np.save('rewards', rewards)
        print(average_time/5)
        average_time = 0

        

    print('Episode: ', eps, '| Episode Reward: ', episode_reward)
    rewards.append(episode_reward)

    eps = eps + 1



