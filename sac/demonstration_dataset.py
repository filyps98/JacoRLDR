import random 
import torch
import os
import numpy as np 
import math

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state_image, state_hand, action, reward, next_state_image, next_state_hand, done):
    
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
        
        if (int((self.position + 1) % self.capacity) == 0):
            print("Enter")
            #the buffer is full
            #Create dataset if it doesn't exist
            if(os.path.isfile("Dataset.pt") == False):
                print("Save")
                torch.save(self.buffer,"Dataset.pt")
                return True
            
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
        
    