#SAVE FILE TO SAY HOW MANY FILES YOU USED IN THE MODEL
import random 
import torch
import os
import numpy as np 
import math


def what_is_next_dataset_index():

    directory = "dataset"
    count = 0

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            count += 1
    
    return count

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state_image, state_hand, action, reward, next_state_image, next_state_hand, done):
    
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
        
        self.position = int((self.position + 1) % self.capacity)
        if (int((self.position) % self.capacity) == 0):
            print("Enter")
            #the buffer is full
            #Create dataset if it doesn't exist
            index = what_is_next_dataset_index() + 1
            torch.save(self.buffer,"dataset/Dataset_"+ str(index))
            
          # as a ring buffer
        
