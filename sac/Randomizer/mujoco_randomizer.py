from Randomizer.body_modifier import Body
from Randomizer.light_randomizer import Light

class Randomizer():
    
    def __init__(self, interface):

        self.model = interface.model
        self.sim = interface.sim

    def body(self,starting_index, number_obj):
        return Body(self.model, starting_index, number_obj)
    
    def light(self):
        return Light(self.sim)