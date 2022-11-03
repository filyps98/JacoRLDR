import numpy as np
import random
from abr_control.utils import transformations

class Body():
    def __init__(self, model,index, number_obj):

        #Body Properties
        self.model = model
        self.starting_index = index
        self.number_obj = number_obj
        self.starting_xyz = [0.35, 0, 0]
        self.starting_mass = 0


    
    def modify_xyz(self, index, range_xyz):
        
        target_xyz = 2*(np.random.rand(3)-0.5) * np.array(range_xyz) + self.starting_xyz
        self.model.body_pos[index] = target_xyz

        return target_xyz
    
    def modify_euler(self, index, range_euler):
        
        target_euler = 2*(np.random.rand(3)-0.5) * np.array(range_euler) 
        target_quat = transformations.quaternion_from_euler(target_euler[0], target_euler[1], target_euler[2], "rxyz")
        self.model.body_quat[index] = target_quat

        return transformations.euler_from_quaternion(target_quat, "rxyz")
    
    def modify_mass(self, index, range_mass):

        target_mass = 2*(np.random.rand(3)-0.5) * np.array(range_mass) + self.starting_mass
        self.model.body_pos[index] = target_mass

        return target_mass
    
    def change_object(self):

        self.reset_position()

        index_list = list(range(self.starting_index, self.starting_index + self.number_obj))
        
        return random.choice(index_list)
    
    def reset_position(self):

        shift = 0.5
        for i in range(self.starting_index, self.starting_index + self.number_obj):
            self.model.body_pos[i] = np.array([1, shift, 0])
            shift = shift - 0.1
        

    
        


