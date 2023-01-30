import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SoftQNetwork(nn.Module):
    def __init__(self, init_w=3e-3):
        #to modify
        super(SoftQNetwork, self).__init__()
        
        #CNN part
        self.batch_norm= nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 2, stride = 1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 2, stride = 1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.pooling2 = nn.MaxPool2d(2)
        self.conv_gen1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 1, stride = 1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size = 3, stride = 2)
        self.conv_gen2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 1, stride = 1)
        self.linear1 = nn.Linear(256,224)

        #Linear Part
        self.linear_1 = nn.Linear(6,64)
        self.linear_2 = nn.Linear(64,64)
        self.linear_3 = nn.Linear(64,32)

        #Linear Combined state
        self.linear_state_combined = nn.Linear(256,512)

        #Linear combined action and state
        self.linear_combined_1= nn.Linear(550,256)
        self.linear_combined_2= nn.Linear(256,128)

        #Linear action
        self.linear_action_1 = nn.Linear(7,64)
        self.linear_action_2 = nn.Linear(64,38)


        self.linear_final = nn.Linear(128,1)
        
        self.linear_combined_2.weight.data.uniform_(-init_w, init_w)
        self.linear_combined_2.bias.data.uniform_(-init_w, init_w)

    def forward_CNN(self,state):

        x = self.batch_norm(state)
        x = F.relu(self.conv1(x))
        x = self.pooling1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv_gen1(x))
        x = self.pooling1(x)
        x = self.batchnorm2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv_gen2(x))
        x = self.pooling2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        return x

    #to modify
    def forward_linear(self, state):
        
        x = F.relu(self.linear_1(state))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))

        return x

    def action_linear(self,action):

        x = F.relu(self.linear_action_1(action))
        x = F.relu(self.linear_action_2(x))

        return x

#to modify  
    def forward(self, state_image, state_hand, action):

        x_image = self.forward_CNN(state_image)
        x_hand = self.forward_linear(state_hand)
        y = self.action_linear(action)
        
        x_image = torch.squeeze(x_image)

        x = torch.cat((x_image,x_hand),1)
        
        x = F.relu(self.linear_state_combined(x))
        
        z = torch.cat((x,y),1)
      
        z = F.relu(self.linear_combined_1(z))
        z = F.relu(self.linear_combined_2(z))
        z = F.relu(self.linear_final(z))
        
        return z
#to modify

