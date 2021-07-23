from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Target(nn.Module):

    def __init__(self, vocab_size, batch_size, embed_dim, hidden_dim,
                 seq_len, start_token,net_paras_save_path,is_load=1):
        super(Target, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.start_token = start_token

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(
            self.embed_dim, self.hidden_dim
        )
        self.fc = nn.Linear(
            self.hidden_dim,
            self.vocab_size
        )
        self._init_params(net_paras_save_path,is_load)

    def _init_params(self,net_paras_save_path,is_load): #need to rewrite for easy usage.
        if (is_load):
            self.load_state_dict(torch.load(net_paras_save_path))
        else:
            for param in self.parameters():
                nn.init.normal(param, std=1.0)

    def init_hidden(self):
        h = Variable(torch.zeros(
            self.batch_size, self.hidden_dim
        ))
        c = Variable(torch.zeros(
            self.batch_size, self.hidden_dim
        ))
        return h, c

    def forward(self, t, x_t, h_t, c_t):
        x_t_embeded = self.embed(x_t)
        h_tp1, c_tp1 = self.recurrent_unit(x_t_embeded, (h_t, c_t))
        logits = self.fc(h_tp1)
        probs = F.softmax(logits, dim=1)
        next_token = Categorical(probs).sample()
        return t + 1, next_token, h_tp1, c_tp1, logits, next_token

def init_vars(net, use_cuda=False):
    h_t, c_t = net.init_hidden()
    x_t = Variable(nn.init.constant(
        torch.LongTensor(net.batch_size), net.start_token
    ))
    vs = [x_t, h_t, c_t]
    s=type(vs)
    if use_cuda:
        for i, v in enumerate(vs):
            v = v.cuda(async=True)
            vs[i] = v
    # return x_t, h_t, c_t
    return vs[0],vs[1],vs[2]

def recurrent_func(f_type='pre',use_cuda=False):

    if f_type == 'pre':
        def func(net, real_data, use_cuda=False):
            '''
            Initialize some variables and lists
            '''
            x_t, h_t, c_t = init_vars(net, use_cuda)
            seq_len = net.seq_len
            logits_list = []

            '''
            Perform forward process.
            '''
            for t in range(seq_len):
                _, _, h_t, c_t, logits, next_token = net(t, x_t, h_t, c_t)
                x_t = real_data[:, t].contiguous()
                logits_list.append(logits)
            logits_var = torch.stack(logits_list).permute(1, 0, 2)
            return logits_var
        return func

    elif f_type == 'gen':
        def func(net, use_cuda=False):
            '''
            Initialize some variables and lists
            '''
            x_t, h_t, c_t = init_vars(net, use_cuda)
            seq_len = net.seq_len
            gen_token_list = []

            '''
            Perform forward process.
            '''
            for t in range(seq_len):
                _, x_t, h_t, c_t, logits, next_token = net(t, x_t, h_t, c_t)
                gen_token_list.append(x_t)
            gen_token_var = torch.stack(gen_token_list).permute(1, 0)
            return gen_token_var
        return func

def nll_loss(net, batch_generated_data, use_cuda=False):
    logits = recurrent_func('pre')(net, batch_generated_data, use_cuda).contiguous()
    batch_size, seq_len = batch_generated_data.size()
    f = nn.CrossEntropyLoss()
    if use_cuda:
        f = f.cuda()
    inputs = logits.view(batch_size * seq_len, -1)
    target = batch_generated_data.view(-1)
    loss = f(inputs, target)
    return loss

def batch_nll_loss(net,dataloader,use_cuda=False):
    loss=0
    for i, sample in enumerate(dataloader):
        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda()
        # optimizer.zero_grad()
        loss+= nll_loss(net, sample, use_cuda)
    return loss


def generate(net, use_cuda=False):
    return recurrent_func('gen')(net, use_cuda)

def generated_train_data(net,batchs,use_cuda=False):
    # batch=generated_number//net.batch_size+1
    train_data=torch.zeros(([0,net.seq_len]),dtype=torch.int64)
    if(use_cuda):
        train_data=train_data.cuda()
    for i in range(batchs):
        generated_one_batch=recurrent_func('gen')(net, use_cuda)
        train_data=torch.cat([train_data,generated_one_batch],dim=0)

    return train_data


def generate_real_sample(net,batchs,filepath,use_cuda=False):
    train_data=generated_train_data(net,batchs,use_cuda)
    train_data_np = train_data.cpu().numpy()
    np.save(filepath, train_data_np)