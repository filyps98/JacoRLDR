from Randomizer.body_modifier import Body
from Randomizer.light_randomizer import Light

class Randomizer():
    
    def __init__(self, interface):

        self.model = interface.model
        self.sim = interface.sim

    def body(self,index):
        return Body(self.model, index)
    
    def light(self):
        return Light(self.sim)