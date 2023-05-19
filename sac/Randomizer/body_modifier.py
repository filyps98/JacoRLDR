import numpy as np
from abr_control.utils import transformations

class Body():
    def __init__(self, model,index):

        #Body Properties
        self.model = model
        self.index = index
        self.starting_xyz = model.body_pos[index].copy()
        self.starting_quat = model.body_quat[index].copy()
        self.starting_mass = model.body_mass[index].copy()

        self.geom_ID = model.body_geomadr[index].copy()
        self.geom_type = model.geom_type[self.geom_ID].copy()
        self.starting_size = model.geom_size[self.geom_ID].copy()
        self.geom_rgba = model.geom_rgba[self.geom_ID].copy()

    
    def modify_xyz(self, range_xyz):
        
        target_xyz = 2*(np.random.rand(3)-0.5) * np.array(range_xyz) + self.starting_xyz
        self.model.body_pos[self.index] = target_xyz

        return target_xyz
    
    def modify_euler(self, range_euler):
        
        target_euler = 2*(np.random.rand(3)-0.5) * np.array(range_euler) 
        target_quat = transformations.quaternion_from_euler(target_euler[0], target_euler[1], target_euler[2], "rxyz") + self.starting_quat
        self.model.body_quat[self.index] = target_quat

        return transformations.euler_from_quaternion(target_quat, "rxyz")
    
    def modify_mass(self, range_mass):

        target_mass = 2*(np.random.rand(3)-0.5) * np.array(range_mass) + self.starting_mass
        self.model.body_pos[self.index] = target_mass

        return target_mass
    
    def modify_size(self, range_size):

        if self.geom_type == 5:
            target_size = 2*(np.random.rand(3)-0.5) * np.array(range_size) + self.starting_size
        
        if self.geom_type == 6:
            target_size = 2*(np.random.rand(3)-0.5) * np.array(range_size) + self.starting_size

        self.model.geom_size[self.geom_ID] = target_size
        
        return target_size
    
    def isolate_object(self):
        self.model.body_pos[self.index] = [1,0,0]

