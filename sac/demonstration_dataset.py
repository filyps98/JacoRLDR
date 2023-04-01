#SAVE FILE TO SAY HOW MANY FILES YOU USED IN THE MODEL
import random 
import torch
import os
import numpy as np 
import math
import datetime


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state_image, state_hand, action, reward, next_state_image, next_state_hand, done):
    
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
        
        # as a ring buffer
        self.position = int((self.position + 1) % self.capacity)
        if (int((self.position) % self.capacity) == 0):
            print("Upload new Dataset")
            #the buffer is full
            torch.save(self.buffer,"dataset/Dataset_"+ str(datetime.datetime.now()))
            
          
        
