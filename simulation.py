"""
Move the jao2 Mujoco arm to a target position.
The simulation ends after 1500 time steps, and the
trajectory of the end-effector is plotted in 3D.
"""
import sys
import traceback
import numpy as np
import glfw
import math
import mujoco_py 
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.special as fun
import wandb

from abr_control.controllers import OSC, Damping, path_planners
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations



class Mujoco_prototype():
    def __init__(self,dir_name,arm_model, vision):

        print("\nSimulation starting...\n")
        # initialize our robot config for the jaco2
        self.robot_config = arm(arm_model, folder = dir_name )

        # create our Mujoco interface
        self.interface = Mujoco(self.robot_config, dt=0.001,visualize = vision, create_offscreen_rendercontext = True)
        self.interface.connect()

        self.start()



    def start(self):

        self.interface.send_target_angles(self.robot_config.START_ANGLES)

        # damp the movements of the arm
        damping = Damping(self.robot_config, kv=10)

        # instantiate controller
        self.ctrlr = OSC(
            self.robot_config,
            kp=30,  # position gain
            kv=20,
            ko=180,  # orientation gain
            null_controllers=[damping],
            vmax= None,  # [m/s, rad/s]
            # control all DOF [x, y, z, alpha, beta, gamma]
            ctrlr_dof=[True, True, True, True, True, True],
        )

        # get the end-effector's initial position
        feedback = self.interface.get_feedback()

        #get initial position and orientation
        self.start_xyz = self.robot_config.Tx("EE", feedback["q"])
        self.start_orientation = self.robot_config.quaternion("EE", feedback["q"])



    def step_sim(self,action, number_step, target_geom_ID, xyz_pos):

        #variable to store the destination
        self.pos_final = np.hstack([action])

        #transform matrix to compute the translation from global coordinates to local right position of the hand
        A = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])

        # get the end-effector's initial position
        feedback = self.interface.get_feedback()

        ee_xyz = self.robot_config.Tx("EE", feedback["q"])
        ee_orientation = self.robot_config.quaternion("EE", feedback["q"])

        #I transform the position of the gripper
        transf = transformations.euler_from_quaternion(ee_orientation, "rxyz") + np.matmul(A,self.pos_final[3:6])
        #position xyz
        position_final = ee_xyz + self.pos_final[:3]

        #Generate the plan, you need the initial position of the gripper at the start of the step and your desired end position
        (
            position_planner,
            orientation_path,
        ) = self.get_target(ee_xyz,ee_orientation, position_final, transf)

        try:
        
            pre_grip = np.copy(self.pos_final[6:])

            for i in range(800):
                
                #generate next step of the path planner
                pos, _ = position_planner.next()
                orient = orientation_path.next()
                self.pos_step = np.hstack([pos, orient,self.pos_final[6:]])
            
                # get joint angle and velocity feedback
                feedback = self.interface.get_feedback()


                # calculate the control signal
                u = self.ctrlr.generate(
                    q=feedback["q"],
                    dq=feedback["dq"],
                    target=self.pos_step[:6],
                )

                # add gripper forces
                if i <= 200:
                    grip = i*(self.pos_step[6:] - pre_grip)/200 + pre_grip
                
                else: grip = self.pos_step[6:]

                # add gripper forces
                u = np.hstack((u, grip))

                # send forces into Mujoco, step the sim forward
                self.interface.send_forces(u)
                        
                # calculate end-effector position
                ee_xyz = self.robot_config.Tx("EE", q=feedback["q"])

                #takes at the end of the movement
                error_pos_int = np.linalg.norm(ee_xyz - position_final)


                #when it reaches a new position save the image
                #even if you don't input anythin at the strt it takes a picture
                #if error_pos_int < error_limit:
                if (error_pos_int < 0.06 and i > 250):

                    next_state_image, next_state_hand = self.get_state()

                    #Evaluate new target position
                    target = self.get_limit_target_pos(target_geom_ID, xyz_pos)

                    #re-get the feedback
                    feedback = self.interface.get_feedback()
                    ee_xyz = self.robot_config.Tx("EE", q=feedback["q"])

                  
                    #Adding distance
                    final_distance = 1/(np.linalg.norm(ee_xyz - target[:3]))
                    reward_from_distance = fun.expit(0.55*final_distance - 4)

                    reward_from_force = 0 
                    reward_from_height = 0
                    

                    if reward_from_distance > 0.8:

                        gripper_sum = np.sum(self.pos_step[6:])
                        reward_gripper = 1 - gripper_sum/24

                        reward_from_force = reward_gripper

                        #Adding Resulting Height
                        #Initialize Resulting Height
                        resulting_height = 0

                        if target[2] < 0.2:
                            resulting_height = target[2]
                        else:
                            resulting_height = 0.2

                        reward_from_height = 10*resulting_height

                    reward = reward_from_distance + reward_from_force + reward_from_height

                    if number_step >= 0:


                        wandb.log({f'Force Reward_{number_step}':reward_from_force})
                        wandb.log({f'Height Reward_{number_step}':reward_from_height})
                        wandb.log({f'Distance Reward_{number_step}':reward_from_distance})
                        wandb.log({f'Force Gripper_{number_step}':np.sum(self.pos_step[6:])})

                        wandb.log({f'Step Reward_{number_step}':reward})

                    return next_state_image, next_state_hand, reward, False

            next_state_image, next_state_hand = self.get_state()
            return next_state_image, next_state_hand, -1, True


        except:
            print("Exception")
            next_state_image, next_state_hand = self.get_state()
            return next_state_image, next_state_hand, -1, True
            


    def get_target(self, ee_position,ee_orientation, target_position, target_orientation):
        # pregenerate our path and orientation planners
        n_timesteps = 200

        position_planner = path_planners.SecondOrderDMP(
            error_scale=0.01, n_timesteps=n_timesteps
        )
        orientation_path = path_planners.Orientation()

        position_planner.generate_path(position=ee_position, target_position=target_position)

        target_orientation = transformations.quaternion_from_euler(target_orientation[0], target_orientation[1], target_orientation[2], "rxyz")

        orientation_path.match_position_path(
            orientation=ee_orientation,
            target_orientation=target_orientation,
            position_path=position_planner.position_path,
        )

        return position_planner, orientation_path

    def get_state(self):

        #image
        image = self.get_image()
        #force = self.get_gripper_forces()
        #force = np.reshape(force, -1)
        pos_hand = self.get_hand_pos()
        
        
        orient_hand = np.array(self.get_hand_orient())
        
        #hand_status = np.concatenate((force, pos_hand, orient_hand), axis = None)

        hand_status = np.concatenate((pos_hand, orient_hand), axis = None)

        return image, hand_status


    def get_image(self):

        self.interface.offscreen._set_mujoco_buffers()
        self.interface.offscreen.render(84,84,0)

        data = self.interface.offscreen.read_pixels(84,84,depth = False)
        image = np.array(data[:,:,:])
        #image = np.array(100*data[1])
        image = np.resize(image,(3,84,84))

        return image
    
    #works in a way I can grasp in a proper way considering the size of the finger
    def get_limit_target_pos(self,target_geom_ID,xyz_pos):

        
        finger_size = 0.08
        half_size_target = self.interface.model.geom_size[target_geom_ID]
        height_target = 0

        if(target_geom_ID == 3):
            xyz_pos[2] = xyz_pos[2] - half_size_target[2] 
            height_target = 2*half_size_target[2]

        elif (target_geom_ID == 4):
            xyz_pos[2] = xyz_pos[2] - half_size_target[1] 
            height_target = 2*half_size_target[1]

        if height_target >= finger_size:
            xyz_pos[2] = xyz_pos[2] + height_target - finger_size

        return xyz_pos


    def get_target_orient(self):
        return transformations.euler_from_quaternion(self.interface.get_orientation("target"), "rxyz")
    
    def get_hand_pos(self):

        feedback = self.interface.get_feedback()
        ee_xyz = self.robot_config.Tx("EE", q=feedback["q"])

        return ee_xyz
    
    def get_hand_orient(self):

        feedback = self.interface.get_feedback()
        ee_orientation = self.robot_config.quaternion("EE", feedback["q"])
        ee_euler = transformations.euler_from_quaternion(ee_orientation, "rxyz")

        return ee_euler


    def restart(self):
        self.interface.sim.reset()
        # get the end-effector's initial position
        self.start()

    def disconnect_sim(self):
        # stop and reset the Mujoco simulation
        self.interface.disconnect()

        print("Simulation terminated...")
