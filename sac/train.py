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
import wandb
import time
import math

from replay import ReplayBuffer
from normalised_action import NormalizedActions
from sac_trainer import SAC_Trainer
from Randomizer import body_swap as bs
from Randomizer.mujoco_randomizer import Randomizer

import argparse

dir_ = os.path.dirname(os.getcwd())

arm_ = "jaco2.xml"
visualize = False
env = Mujoco_prototype(dir_,arm_, visualize)


wandb.init(config = {"algorithm": "JacoRL2"}, project="JacoRL2", entity="pippo98")


replay_buffer_size = 5e4
replay_buffer = ReplayBuffer(replay_buffer_size)

action_dim = 7
action_range = 1

# hyper-parameters for RL training
max_episodes  = 5000000
max_steps = 5


frame_idx   = 0
batch_size  = 64
explore_steps = 0  # for random action sampling in the beginning of training
initial_update_itr = 5
update_itr = 5
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
rewards     = []
model_path = './model/sac_v2'

sac_trainer=SAC_Trainer(replay_buffer, action_dim, action_range=action_range )

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

path_summary = model_path+"\_trained_dataset.txt"
path_dataset = "dataset"

if (os.path.exists(model_path+"\_pre_q1") and os.path.exists(model_path+"\_pre_q2")):
    sac_trainer.load_pre_trained_model(model_path)
    replay_buffer.select_datasets(path_summary)

else:
    with open(path_summary, "w") as file:
        file.write("Following all the dataset that the pre-policy model was trained on:")


while(replay_buffer.load_next_dataset(path_summary)):
    # pre-training loop

    for i in range(1, math.ceil(replay_buffer_size/batch_size)):
        for i in range(initial_update_itr):
            _=sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-0.1*action_dim, train_policy = False)

sac_trainer.save_pre_trained_model(model_path)

geom_body_ID, target_pos, target_orient, size = bs.body_swap(body_cube, body_cylinder)


for eps in range(max_episodes):

    
    #randomize the position and orientation every step 
    env.restart()

    #randomize position
    if(eps%20 == 0):
        geom_body_ID, target_pos, target_orient, size = bs.body_swap(body_cube, body_cylinder)
        light._rand_textures()
        light._rand_lights()
    
    target_state, z_height, max_dimension = env.get_limit_target_pos(geom_body_ID)
    state_image, state_hand = env.get_state(target_state)
    episode_reward = 0


    #I don't want to be too close by the target
    #target_estimated_pos = (target_pos + np.array([0 , 0 , 0.1])).tolist()
    target_estimated_pos = target_pos + (np.array([0.1 , 0.1, 0]*(np.random.rand(3)-0.5)+np.array([0 , 0 , 0.15]))).tolist()
    #target_estimated_orientation = list(target_orient)
    target_estimated_orientation = [0, 0, 0]
    initial_gripper_force = [5,5,5]

    #I initialize and resize the first action

    #estimate how much to shift
    #shifted_xyz = [0,0,0.15]
    shifted_xyz = target_estimated_pos - env.get_hand_pos() 
    scripted_action = np.array([shifted_xyz, target_estimated_orientation, initial_gripper_force])
    scripted_action = np.resize(scripted_action,(9))
    

    _, _, _, _ = env.step_sim(scripted_action, -1, max_steps,  geom_body_ID)

    action = np.zeros(9)


    for step in range(max_steps):

        if frame_idx > explore_steps:
            #get action from the network
            action_RL = sac_trainer.policy_net.get_action(state_image, state_hand, deterministic = DETERMINISTIC)
        
        else:
            #sample action from a distribution
            action_RL = sac_trainer.policy_net.sample_action()
        

        #I add the previous action with a small RL action
        action[:6] = np.multiply(ratio_, action_RL[:6])

        #except the fore grippers that have a greater action space
        action[6:] = ratio_residual_force*action_RL[6]*np.ones(3) + ratio_residual_force


        next_state_image, next_state_hand, reward, done = env.step_sim(action, step , max_steps, geom_body_ID)       
        
            
        replay_buffer.push(state_image, state_hand, action_RL, reward, next_state_image, next_state_hand, done)

        #the next state is now the new state
        state_image = next_state_image
        state_hand = next_state_hand
        
        #updating episode reward
        episode_reward += reward
        frame_idx += 1

        
        if len(replay_buffer) > batch_size:
            for i in range(update_itr):
                _=sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-0.8*action_dim, train_policy = True)

        if done:
            break

    average_rewards = average_rewards + episode_reward

    if eps % 20 == 0 and eps>0: # plot and model saving interval
        wandb.log({"episode reward every_5":average_rewards/20})
        average_rewards = 0
        np.save('rewards', rewards)
        sac_trainer.save_model(model_path)
        

    print('Episode: ', eps, '| Episode Reward: ', episode_reward)
    rewards.append(episode_reward)
sac_trainer.save_model(model_path)



