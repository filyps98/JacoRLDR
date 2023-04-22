import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.distributions import Normal

#to modify
class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, _device, pre_model, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.device = _device
        self.feature_extractor = pre_model
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #CNN partx = F.relu(self.conv_gen2(x))
        self.batch_norm_1= nn.BatchNorm2d(256)
        self.batch_norm_2= nn.BatchNorm2d(512)
        self.conv_gen1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 1, stride = 1)
        self.conv_gen2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size = 3, stride = 2)
        self.conv_gen3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size = 1, stride = 1)
        self.conv_gen4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size = 3, stride = 2)
        self.pooling1 = nn.MaxPool2d(kernel_size= 2, stride = 2)

        self.linear1 = nn.Linear(4096,480)

        #Linear Part
        self.linear_1 = nn.Linear(6,64)
        self.linear_2 = nn.Linear(64,64)
        self.linear_3 = nn.Linear(64,32)

        #Linear Combined

        self.linear_combined_1 = nn.Linear(512,1024)
        self.linear_combined_2 = nn.Linear (1024,1024)
        self.linear_combined_3 = nn.Linear(1024,512)

        self.mean_linear = nn.Linear(512, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(512, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward_CNN(self, state):

        x = self.batch_norm_1(state)
        x = F.relu(self.conv_gen1(x))
        x = self.batch_norm_1(x)
        x = F.relu(self.conv_gen1(x))
        x = self.batch_norm_1(x)
        x = F.relu(self.conv_gen2(x))
        x = self.pooling1(x)
        x = self.batch_norm_2(x)
        x = F.relu(self.conv_gen3(x))
        x = self.batch_norm_2(x)
        x = F.relu(self.conv_gen3(x))
        x = self.pooling1(x)
        x = self.batch_norm_2(x)
        x = F.relu(self.conv_gen4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        return x

#to modify
    def forward_Linear(self, state):

        x = F.relu(self.linear_1(state))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))

        return x

#to modify
    def forward(self, state_image, state_hand):

        x_cnn = self.forward_CNN(state_image)
        x_lin = self.forward_Linear(state_hand)
        
        x = torch.cat((x_cnn,x_lin),1)
        
        
        x = self.linear_combined_1(x)
        x = self.linear_combined_2(x)
        x = self.linear_combined_2(x)
        x = self.linear_combined_3(x)

        mean    = (self.mean_linear(x))

        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state_image, state_hand, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''

        mean, log_std = self.forward(state_image, state_hand)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state_image, state_hand, deterministic):

        state_image = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)

        feature_state = self.feature_extractor(state_image)
        state_hand = torch.FloatTensor(state_hand).unsqueeze(0).to(self.device)


        mean, log_std = self.forward(feature_state, state_hand)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(self.device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        
        action = self.action_range* torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()