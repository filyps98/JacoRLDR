import os
import sys


sys.path.append('../')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from simulation import Mujoco_prototype

import numpy as np
import wandb
import time

from replay import ReplayBuffer
from normalised_action import NormalizedActions
from sac_trainer import SAC_Trainer
from demonstration import scripted_policy
from Randomizer import body_swap as bs
from Randomizer.mujoco_randomizer import Randomizer

import argparse

dir_ = os.path.dirname(os.getcwd())

arm_ = "jaco2.xml"
visualize = True
env = Mujoco_prototype(dir_,arm_, visualize)

wandb.init(config = {"algorithm": "JacoRL_test_time"}, project="JacoRL_test_time", entity="pippo98")


replay_buffer_size = 2e5
replay_buffer = ReplayBuffer(replay_buffer_size)

action_dim = 7
action_range = 1

# hyper-parameters for RL training
max_episodes  = 500000
max_steps = 5

frame_idx   = 0
batch_size  = 550
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
rewards     = []
model_path = './model/sac_v2'

sac_trainer=SAC_Trainer(replay_buffer, action_dim, action_range=action_range )

average_rewards = 0

#Action range for each action
ratio_xy = 0.1
ratio_orient = 0.3
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


sac_trainer.load_model(model_path)

# training loop
    

for eps in range(10):
    
    #randomize the position and orientation every step 
    env.restart()

    #randomize position
    geom_body_ID, target_pos, target_orient, size = bs.body_swap(body_cube, body_cylinder)

    light._rand_textures()
    light._rand_lights()

    state_image, state_hand = env.get_state()
    episode_reward = 0

    

    #I don't want to be too close by the target
    #target_estimated_pos = (target_pos + np.array([0 , 0 , 0.1])).tolist()
    target_estimated_pos = (target_pos + 0.05*np.random.rand(3)+np.array([0 , 0 , 0.2])).tolist()
    target_estimated_orientation = list(target_orient)
    #target_estimated_orientation = [0, 0, 0]
    initial_gripper_force = [5,5,5]

    #I initialize and resize the first action

    #estimate how much to shift
    shifted_xyz = target_estimated_pos - env.get_hand_pos() 
    scripted_action = np.array([shifted_xyz, target_estimated_orientation, initial_gripper_force])
    scripted_action = np.resize(scripted_action,(9))
    

    _, _, _, _ = env.step_sim(scripted_action, -1,  geom_body_ID)


    action = np.zeros(9)

    for step in range(max_steps):
        action_RL = sac_trainer.policy_net.get_action(state_image, state_hand, deterministic = DETERMINISTIC)

        action[:6] = np.multiply(ratio_, action_RL[:6])

        #except the fore grippers that have a greater action space
        action[6:] = ratio_residual_force*action_RL[6]*np.ones(3) + ratio_residual_force

        next_state_image, next_state_hand, reward, done = env.step_sim(action, step, geom_body_ID) 


        episode_reward += reward
        state_image = next_state_image
        state_hand = next_state_hand

    print('Episode: ', eps, '| Episode Reward: ', episode_reward)