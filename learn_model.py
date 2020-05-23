import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True) #**XAVIER init
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = torch.sigmoid(self.linear1(x))  #**sigmoid activation to help guard against exploding gradients!
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.tanh(self.linear4(x))

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(Actor, self).__init__()
        #self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        #x = self.batch_norm(state)
        x = torch.sigmoid(self.linear1(state))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.tanh(self.linear4(x)) #*TAN to scale us to the scores range!
        #throttle_check = 0
        #explode_check = x.tolist()
        #explode_check = explode_check[0]
        #for idx in explode_check:
        #    if idx == -1 or idx == 1:
        #        throttle_check += 1
        #if throttle_check >= 1:
        #    print("GRADIENTS EXPLODED. STOPPING TRAINING ITS USELESS NOW. YOU ARE IN TROUBLE")
        #    import sys
        #    sys.exit(1)
        return x