from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import  numpy as np
import torch.optim as optim

class Worker(nn.Module):
    def __init__(self):
        super(Worker, self).__init__()
        self.goal_init = nn.Parameter(torch.ones(1, 1))

    def forward(self, input):
        # input=input.detach() #加了这句manager没了grad，worker一直有
        output=torch.matmul(self.goal_init,input)
        return output
class Manager(nn.Module):
    def __init__(self):
        super(Manager, self).__init__()
        self.pinit = nn.Parameter(torch.ones(1, 1))

    def forward(self, input):
        input=input.detach()          #detach == op(set is_leaf=True.)
        output=torch.matmul(self.pinit,input)
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.worker=Worker()
        self.manager=Manager()

    def forward(self, input):
        moutput=self.manager(input)
        output=self.worker(moutput)
        return output

w_lr=m_lr=0.1
generator=Generator()
worker=generator.worker
manager=generator.manager
w_optimizer = optim.Adam(worker.parameters(), lr=w_lr)
m_optimizer = optim.Adam(manager.parameters(), lr=m_lr)

input = torch.tensor([5], dtype=torch.float32)
target =torch.tensor([2], dtype=torch.float32)



losslist=[]
for i in range(100):
    output = generator(input)
    input1=output


    # loss = output-target
    # torch.autograd.backward(loss,generator.parameters(),retain_graph=True)
    #
    # w_optimizer.zero_grad()
    # m_optimizer.zero_grad()
    # torch.autograd.backward(loss,worker.parameters(),retain_graph=True)
    #
    # w_optimizer.zero_grad()
    # m_optimizer.zero_grad()
    # torch.autograd.backward(loss,manager.parameters(),retain_graph=False)
    #
    # w_optimizer.zero_grad()
    # m_optimizer.zero_grad()


#---------------------------------------
    output1=generator(input1)
    loss=(output1-target)
    torch.autograd.backward(loss,generator.parameters(),retain_graph=True)
    a=3
    #output=5 x para
    #output1 = 5 x para x para
    #no detach()   wgrad=10  mgrad=10

    # output=5 x para ,with para(1) detach , output=5
    # output1 =5  x para
    #manager detacch  wgrad=5 magrad=5

#---------------------------------------

output = generator(input)
a=3
