import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.batch_norm= nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size = 2, stride = 1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 2, stride = 1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.pooling2 = nn.MaxPool2d(2)
        self.conv_gen1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 1, stride = 1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size = 3, stride = 2)
        self.conv_gen2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 1, stride = 1)
        self.linear1 = nn.Linear(256,512)
        self.linear2 = nn.Linear(512, 1)
        # weights initialization
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):

        #image_state = state[0]
        x = self.batch_norm(x)
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
        x = self.linear2(x)
        
        return x
        