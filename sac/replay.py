import random 
import numpy as np
import os
import torch

#it counts the number of dataset in the dataset directory
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

        self.index = 0

        #It stores the name of all the datasets we are going to access
        self.list_dataset = []
        for filename in os.listdir("dataset"):
            self.list_dataset.append(filename)
    
    #loading next dataset
    def load_next_dataset(self):
        
        #goes throgh all dataset until it doesn't and it resets the buffer to 0
        if(self.index < len(self.list_dataset)):
            self.buffer = torch.load("dataset/"+self.list_dataset[self.index]).copy()
            self.index = self.index + 1
            return True
        
        else:
            print("no more datasets")
            self.buffer = []
            return False
    
    #Creates other datasets
    def push(self, state_image, state_hand, action, reward, next_state_image, next_state_hand, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_image, state_hand, action, reward, next_state_image, next_state_hand, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
        
        if (int((self.position) % self.capacity) == 0):
            print("Enter")
            #the buffer is full
            #Create dataset if it doesn't exist
            index = what_is_next_dataset_index() + 1
            torch.save(self.buffer,"dataset/Dataset_"+ str(index))

        
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_image, state_hand, action, reward, next_state_image, next_state_hand, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state_image, state_hand, action, reward, next_state_image, next_state_hand, done
    
    def __len__(self):
        return len(self.buffer)
    