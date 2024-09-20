import torch.nn as nn
import torch
    
class Critic_Intersection(nn.Module):
    def __init__(self, obs_size, actions_size=0):
        super(Critic_Intersection, self).__init__()
        self.line = nn.Sequential(
            nn.Linear(obs_size+actions_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 3)
        )
    def forward(self, x, acts=None):  
        if acts != None: x = self.line(torch.cat((x, acts), -1))
        else: x = self.line(x)
        return x
    
class Critic_OC(nn.Module):
    def __init__(self, obs_size, acts_size=0):
        super(Critic_OC, self).__init__()
        self.line = nn.Sequential(
            nn.Linear(obs_size+acts_size, 2048),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 6)
        )   
    def forward(self, x, acts=None):  
        if acts != None: x = self.line(torch.cat((x, acts), -1))
        else: x = self.line(x)
        return x
    
class Critic_Gems(nn.Module):
    def __init__(self, acts_size=0):
        super(Critic_Gems, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.line = nn.Sequential(
            nn.Linear(5184+acts_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 5)
        )   

    def forward(self, x, acts=None):  
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        if acts != None: x = self.line(torch.cat((x, acts), -1))
        else: x = self.line(x)
        return x
    