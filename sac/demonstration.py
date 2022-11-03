import numpy as np


def randomize():
    return 2*(np.random.rand(6) - 0.5) * np.array([0.1, 0.1, 0.1, 0, 0, 0]) + np.array([0.3, 0.005, 0.25, 0, 0, 0])

def scripted_policy(n_step):

    #actions that must not be randomized
    final_position = np.zeros((6,6))
    final_position[0] = [0.3, 0.01, 0.25, 10, 10, 10]
    #final_position[0] = randomize()
    final_position[1] = [0.3, 0.007, 0.17, 10, 10, 10]
    #final_position[1] = randomize()
    final_position[2] = [0.3, 0.006, 0.13, 10, 10, 10]
    #final_position[2] = randomize()
    final_position[3] = [0.3, 0.005, 0.008, 10, 10, 10]
    final_position[4] = [0.3, 0.005, 0.008, 0.4, 0.4, 0.4]
    final_position[5] = [0.3, 0.005, 0.15, 0.4, 0.4, 0.4]

    return final_position[n_step]