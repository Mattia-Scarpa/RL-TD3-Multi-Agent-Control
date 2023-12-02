import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,fc1,fc2, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1_1=nn.Linear(state_size,fc1)
        self.fc2_1=nn.Linear(fc1+action_size,fc2)
        
        self.bn_1=nn.BatchNorm1d(state_size)
        self.bn2_1=nn.BatchNorm1d(fc1)
 
        self.fc5_1=nn.Linear(fc2,1)
        
        #last layer weight and bias initialization 
        self.fc5_1.weight.data.uniform_(-3e-4, 3e-4)
        self.fc5_1.bias.data.uniform_(-3e-4, 3e-4)
        
        
        self.fc1_2=nn.Linear(state_size,fc1)
        self.fc2_2=nn.Linear(fc1+action_size,fc2)

        self.bn_2=nn.BatchNorm1d(state_size)
        self.bn2_2=nn.BatchNorm1d(fc1)

        self.fc5_2=nn.Linear(fc2,1)

        #last layer weight and bias initialization
        self.fc5_2.weight.data.uniform_(-3e-4, 3e-4)
        self.fc5_2.bias.data.uniform_(-3e-4, 3e-4)
        
    def forward(self, state,action):
        """Build a network that maps state & action to action values Q1 and Q2."""
        
        x_1=self.bn(state)
        x_1=F.relu(self.bn2(self.fc1(x_1)))
        x_1=torch.cat([x_1,action],1)
        x_1=F.relu(self.fc2(x_1))
        
        x_1=self.fc5(x_1)

        x_2=self.bn_2(state)
        x_2=F.relu(self.bn2_2(self.fc1(x_2)))
        x_2=torch.cat([x_2,action],1)
        x_2=F.relu(self.fc2_2(x_2))
        
        x_2=self.fc5_2(x_2)
        
        return x_1, x_2
    
    def Q1(self, state,action):
        """Forward pass for Q1"""
        x_1=self.bn_1(state)
        x_1=F.relu(self.bn2_1(self.fc1_1(x_1)))
        x_1=torch.cat([x_1,action],1)
        x_1=F.relu(self.fc2_1(x_1))
        
        x_1=self.fc5_1(x_1)
        
        return x_1

    
class Actor(nn.Module):

    def __init__(self,state_size, action_size, fc1,fc2,seed):
        super(Actor, self).__init__()
        

        # network mapping state to action 

        self.seed = torch.manual_seed(seed)
        
        self.bn=nn.BatchNorm1d(state_size)
        self.bn2=nn.BatchNorm1d(fc1)
        self.bn3=nn.BatchNorm1d(fc2)
        
        self.fc1= nn.Linear(state_size,fc1)
        self.fc2 = nn.Linear(fc1,fc2)
        self.fc4 = nn.Linear(fc2, action_size)
        
        #last layer weight and bias initialization 
        torch.nn.init.uniform_(self.fc4.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.fc4.bias, a=-3e-3, b=3e-3)
        
        # Tanh
        self.tan = nn.Tanh()
        
        
    def forward(self, x):

        x=self.bn(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))

        return self.tan(self.fc4(x))

    