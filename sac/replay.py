import random 
import numpy as np
import os
import torch
import datetime


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        self.index = 0

        #It stores the name of all the datasets we are going to access
        self.list_dataset = []
        
    
    #loading next dataset
    def load_next_dataset(self, path_summary):
        
        #goes throgh all dataset until it doesn't and it resets the buffer to 0
        if(self.index < len(self.list_dataset)):

            print("Enter in: " + self.list_dataset[self.index])

            self.buffer = torch.load("dataset/"+self.list_dataset[self.index]).copy()

            with open(path_summary, "a") as file:
                file.write("\n" + self.list_dataset[self.index])

            self.index = self.index + 1

            

            return True
        
        else:
            print("No more datasets to load and train")
            self.buffer = []
            return False
    
    #Creates other datasets
    def push(self, state_image, state_hand, action, reward, next_state_image, next_state_hand, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_image, state_hand, action, reward, next_state_image, next_state_hand, done)

        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
        
        if (int((self.position) % self.capacity) == 0):
            print("Upload new dataset")
            #the buffer is full
            #Create dataset if it doesn't exist
            torch.save(self.buffer,"dataset/Dataset_"+ str(datetime.datetime.now()))

        
    
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
    
    def select_datasets(self, path_summary):

        # read in the names of files from a text file
        with open(path_summary, "r") as file:
            file_names = [line.strip() for i, line in enumerate(file) if i > 0]

        # create an empty array to store the names of missing files
        self.list_dataset = []

        # check if each file exists in the directory
        for file_name in os.listdir("dataset"):
            
            if file_name not in file_names:
                self.list_dataset.append(file_name)

        
    