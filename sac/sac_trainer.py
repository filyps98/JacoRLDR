import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from soft_network import SoftQNetwork
from policy_network import PolicyNetwork

class SAC_Trainer():
    def __init__(self, replay_buffer, action_dim, action_range):

        GPU = True
        device_idx = 0
        if GPU:
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(self.device)
        
        self.vgg16 = models.vgg16(pretrained=True)
        print(self.vgg16)
        self.feature_extractor = nn.Sequential(*list(self.vgg16.features.children())[:12])

        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork().to(self.device)
        self.soft_q_net2 = SoftQNetwork().to(self.device)
        self.target_soft_q_net1 = SoftQNetwork().to(self.device)
        self.target_soft_q_net2 = SoftQNetwork().to(self.device)
        self.policy_net = PolicyNetwork(action_dim, self.device, self.feature_extractor, action_range).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state_image, state_hand, action, reward, next_state_image, next_state_hand, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state_image = torch.FloatTensor(state_image).to(self.device)
        state_hand = torch.FloatTensor(state_hand).to(self.device)
        next_state_image = torch.FloatTensor(next_state_image).to(self.device)
        next_state_hand = torch.FloatTensor(next_state_hand).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        feature_state =  self.feature_extractor(state_image)
        feature_next_state = self.feature_extractor(next_state_image)

        predicted_q_value1 = self.soft_q_net1(feature_state, state_hand, action)
        predicted_q_value2 = self.soft_q_net2(feature_state,state_hand, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(feature_state, state_hand)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(feature_next_state, next_state_hand)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        wandb.log({"alpha":self.alpha})
        wandb.log({"alpha_loss":alpha_loss})


    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(feature_next_state, next_state_hand, new_next_action),self.target_soft_q_net2(feature_next_state, next_state_hand, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward(retain_graph = True)
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), 5)
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward(retain_graph = True)
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), 5)
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(feature_state, state_hand, new_action),self.soft_q_net2(feature_state, state_hand, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()


        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        wandb.log({"loss":policy_loss})


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1', map_location=torch.device('cpu')))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2', map_location=torch.device('cpu')))
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location=torch.device('cpu')))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
