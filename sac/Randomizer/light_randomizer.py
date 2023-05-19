import mujoco_py
import numpy as np

from mujoco_py.modder import CameraModder, LightModder

from Randomizer.utils.modder import TextureModder
from Randomizer.utils.sim import Range, Range3D
from Randomizer.utils.sim import sample, sample_xyz, sample_light_dir

class Light():

    def __init__(self, sim):

        self.sim = sim
        self.model = self.sim.model
        self.tex_modder = TextureModder(self.sim)
        #self.tex_modder.whiten_materials()  # ensures materials won't impact colors
        self.cam_modder = CameraModder(self.sim)
        self.light_modder = LightModder(self.sim)


    def _rand_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        bright = np.random.binomial(1, 0.5)

        object_ = [self.model.geom_names[0], self.model.geom_names[1],self.model.geom_names[19]]

        for name in object_:
            self.tex_modder.rand_all(name)
            if bright: 
                self.tex_modder.brighten(name, np.random.randint(0,150))

    def _rand_lights(self):
        """Randomize pos, direction, and lights"""
        # light stuff
        #X = Range(-1.5, 1.5) 
        #Y = Range(-1.2, 1.2)
        #Z = Range(0, 2.8)
        X = Range(-1.5, 1.5) 
        Y = Range(-1.2, 1.2)
        Z = Range(0, 2.8)
        LIGHT_R3D = Range3D(X, Y, Z)
        LIGHT_UNIF = Range3D(Range(0,1), Range(0,1), Range(0,1))

        # TODO: also try not altering the light dirs and just keeping them at like -1, or [0, -0.15, -1.0]
        for i, name in enumerate(self.model.light_names):
            lid = self.model.light_name2id(name)
            # random sample 80% of any given light being on 
            if lid != 0:
                self.light_modder.set_active(name, sample([0,1]) < 0.8)
                self.light_modder.set_dir(name, sample_light_dir())

            self.light_modder.set_pos(name, sample_xyz(LIGHT_R3D))

            #self.light_modder.set_dir(name, sample_xyz(rto3d([-1,1])))

            #self.light_modder.set_specular(name, sample_xyz(LIGHT_UNIF))
            #self.light_modder.set_diffuse(name, sample_xyz(LIGHT_UNIF))
            #self.light_modder.set_ambient(name, sample_xyz(LIGHT_UNIF))

            spec =    np.array([sample(Range(0.5,1))]*3)
            diffuse = np.array([sample(Range(0.5,1))]*3)
            ambient = np.array([sample(Range(0.5,1))]*3)

            self.light_modder.set_specular(name, spec)
            self.light_modder.set_diffuse(name,  diffuse)
            self.light_modder.set_ambient(name,  ambient)
            #self.model.light_directional[lid] = sample([0,1]) < 0.2
            self.model.light_castshadow[lid] = sample([0,1]) < 0.5
    