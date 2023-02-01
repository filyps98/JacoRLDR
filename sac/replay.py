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
        self.buffer_complete = False
        self.isthereDataset = False
        self.buffer_combined = []
        self.final_partition = 0.7

        if(os.path.isfile("Dataset.pt") == False):
            self.buffer_combined = self.buffer
        else:
            self.dataset = torch.load("Dataset.pt").copy()
            
            if(len(self.dataset) >= self.capacity):
                self.buffer_combined  = random.sample(self.dataset,self.capacity)
                self.isthereDataset = True
            
            else:
                self.buffer_combined = self.buffer


    
    def push(self, state_image, state_hand, action, reward, next_state_image, next_state_hand, done):
    
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
        
        if (int((self.position + 1) % self.capacity) == 0):

            #the buffer is full
            self.buffer_complete = True
            #Create dataset if it doesn't exist
            if(os.path.isfile("Dataset.pt") == False):
        	    torch.save(self.buffer,"Dataset.pt")
            else:
                #Or add it in the Queue
                torch.save(torch.load("Dataset.pt") + self.buffer,"Dataset.pt")
        	
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
        
    
    def sample(self, batch_size):
    
    	ratio = int(math.floor((self.position+1)/self.capacity))*100
    	
        if(ratio%10 == 0 and self.isthereDataset):
    	
            if (self.buffer_complete != True):
        
                batch_dataset = random.sample(self.dataset,self.capacity - self.position)
                self.buffer_combined = self.buffer + batch_dataset
            
            else: 
            
                batch_partition = math.floor(int((self.capacity)*self.final_partition))
                self.buffer_combined = random.sample(self.buffer, batch_partition) + random.sample(self.dataset, self.capacity - batch_partition)
	

        batch = random.sample(self.buffer_combined, batch_size)
        state_image, state_hand, action, reward, next_state_image, next_state_hand, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state_image, state_hand, action, reward, next_state_image, next_state_hand, done
    
    def __len__(self):
        return len(self.buffer_combined)
