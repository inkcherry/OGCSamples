import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.autograd import Variable
from torch.distributions import Categorical


# A truncated distribution has its domain (the x-values) restricted to a certain range of values. For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated


class Manager(nn.Module):
    def __init__(self, batch_size, hidden_dim, goal_out_size):
        super(Manager, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.recurrent_unit = nn.LSTMCell(
            self.goal_out_size,  # input size
            self.hidden_dim  # hidden size
        )
        self.fc = nn.Linear(
            self.hidden_dim,  # in_features
            self.goal_out_size  # out_features
        )

        self.goal_init = nn.Parameter(torch.zeros(self.batch_size, self.goal_out_size))
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
        self.goal_init.data = truncated_normal(
            self.goal_init.data.shape
        )

    def forward(self, f_t, h_m_t, c_m_t):
        """
        f_t = feature of CNN from discriminator leaked at time t, it is input into LSTM
        h_m_t = ouput of previous LSTMCell
        c_m_t = previous cell state
        """
        # print("H_M size: {}".format(h_m_t.size()))
        # print("C_M size: {}".format(c_m_t.size()))
        # print("F_t size: {}".format(f_t.size()))
        # f_t=f_t.cuda(0) #detach in dis

        h_m_tp1, c_m_tp1 = self.recurrent_unit(f_t, (h_m_t, c_m_t))
        sub_goal = self.fc(h_m_tp1)
        # sub_goal = torch.renorm(sub_goal, 2, 0,
        #                         1.0)   renorm dim=0 ===normalize dim=1

        sub_goal=F.normalize(sub_goal,p=2,dim=1)

        # Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
        return sub_goal, h_m_tp1, c_m_tp1


class Worker(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_dim, hidden_dim,
                 goal_out_size, goal_size):
        super(Worker, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size + 48 + 16  # 64pos 16keep/blank

        # self.emb = nn.Embedding(self.vocab_size, self.embed_dim)

        self.emb = nn.Embedding(self.vocab_size, self.embed_dim,padding_idx=32)
        self.recurrent_unit = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        # self.fc = nn.Linear(self.hidden_dim+50, (self.goal_size+16)*self.vocab_size)
        self.fc = nn.Linear(self.hidden_dim +80, self.goal_size * self.vocab_size)
        # self.fc_z = nn.Linear(1000, 50)  # tag for melody_z
        # tag  for melody_z
        # self.goal_change = nn.Parameter(torch.zeros(self.goal_out_size, self.goal_size+16))
        self.goal_change = nn.Parameter(torch.zeros(self.goal_out_size, self.goal_size))
        self._init_params()
        # self.posfc = nn.Linear(self.)

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)

    def forward(self, x_t, h_w_t, c_w_t, pos_vec, melody_z):
        """
            x_t = last word
            h_w_t = last output of LSTM in Worker
            c_w_t = last cell state of LSTM in Worker
        """

        x_t_emb = self.emb(x_t)
        h_w_tp1, c_w_tp1 = self.recurrent_unit(x_t_emb, (h_w_t, c_w_t))
        # melody_goal = self.fc_z(melody_z)
        output_tp0 = torch.cat([h_w_tp1,pos_vec], dim=1)

        output_tp1 = self.fc(output_tp0)
        # output_tp1 = output_tp1.view(self.batch_size, self.vocab_size, self.goal_size+16)
        output_tp1 = output_tp1.view(self.batch_size, self.vocab_size, self.goal_size)
        #
        # pos_vec = pos_vec.unsqueeze(2)
        #
        # output_tp1 = torch.mul(pos_vec, output_tp1.permute(0, 2, 1))
        # output_tp1 = output_tp1.permute(0, 2, 1)
        return output_tp1, h_w_tp1, c_w_tp1


class Generator(nn.Module):
    def __init__(self, worker_params, manager_params, step_size):
        super(Generator, self).__init__()
        self.step_size = step_size
        self.worker = Worker(**worker_params)
        self.manager = Manager(**manager_params)

        self.beat_pos = torch.FloatTensor(self.worker.batch_size, 4).cuda()
        self.bar_pos = torch.FloatTensor(self.worker.batch_size, 8).cuda()
        self.pixel_pos = torch.FloatTensor(self.worker.batch_size, 4).cuda()

        self.keep_time = torch.FloatTensor(self.worker.batch_size, 8).cuda()
        self.blank_time = torch.FloatTensor(self.worker.batch_size, 8).cuda()

    def init_hidden(self):
        h = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        c = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        return h, c

    def init_melody_z(self):  # batch x dimension
        self.melody_z = truncated_normal([self.worker.batch_size, 1000], lower=-1, upper=1).cuda()
        self.keep_time_num = torch.zeros([self.worker.batch_size, 1], dtype=torch.long).cuda()
        self.blank_time_num = torch.zeros([self.worker.batch_size, 1], dtype=torch.long).cuda()
        return

    def forward(self, x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, t, temperature, is_gen=False):
        # f_t = f_t.detach() #detach in dis
        x_t = x_t.detach()
        # detach hidden state
        #
        # h_w_t = h_w_t.detach()
        # c_w_t = c_w_t.detach()
        #
        # h_m_t = h_m_t.detach()
        # c_m_t = c_m_t.detach()
        #
        sub_goal, h_m_tp1, c_m_tp1 = self.manager(f_t, h_m_t, c_m_t)

        is_keep = (x_t[:] == 15)
        is_blank = (x_t[:] == 16)
        self.keep_time_num = is_keep.unsqueeze(1).long().mul(self.keep_time_num[:, :] + 1).clamp_max_(7)
        self.blank_time_num = is_blank.unsqueeze(1).long().mul(self.blank_time_num[:, :] + 1).clamp_max_(7)
        # mmm=self.keep_time_num
        if (t < 128):
            # does it need pixel pos?
            beat_pos_num = torch.LongTensor(self.worker.batch_size, 1).fill_(t % 16 // 4).cuda()
            bar_pos_num = torch.LongTensor(self.worker.batch_size, 1).fill_(t // 16).cuda()
            pixel_pos_num = torch.LongTensor(self.worker.batch_size, 1).fill_(t % 16 % 4).cuda()


            self.keep_time.fill_(0).scatter_(1, self.keep_time_num, 1).cuda()
            self.blank_time.fill_(0).scatter_(1, self.blank_time_num, 1).cuda()



            # tt = torch.FloatTensor(self.worker.batch_size, 2).cuda().fill_(1)

            self.beat_pos.zero_().scatter_(1, beat_pos_num, 1).cuda()
            self.bar_pos.zero_().scatter_(1, bar_pos_num, 1).cuda()
            self.pixel_pos.zero_().scatter_(1, pixel_pos_num, 1).cuda()

            # self.pos_vec = torch.cat([self.beat_pos, self.pixel_pos, self.bar_pos,self.keep_time,self.blank_time], 1).cuda()
            self.pos_vec = torch.cat([self.beat_pos, self.pixel_pos, self.bar_pos], 1).cuda()
            # expand 4 time
            self.pos_vec_cat = torch.cat(
                [self.pos_vec, self.pos_vec, self.pos_vec, self.pos_vec, self.keep_time, self.blank_time], 1)

        output, h_w_tp1, c_w_tp1 = self.worker(x_t, h_w_t, c_w_t, self.pos_vec_cat, self.melody_z)

        last_goal_temp = last_goal + sub_goal
        # tag no gradient
        # check=real_goal.cpu().detach().numpy()

        # final_goal=torch.cat([real_goal.detach(),melody_goal],dim=1)

        # kn=torch.ones([64,1720],dtype=torch.float32).cuda()

        w_t = torch.matmul(
            real_goal.detach(), self.worker.goal_change
        )




        # w_t = torch.renorm(w_t, p=2, dim=0,maxnorm= 1.0)  #64 * 80 #renorm dim=0 === F.normalization dim=1

        w_t = F.normalize(w_t, p=2, dim=1)

        w_t = torch.unsqueeze(w_t, -1)

        # if(t>=127):
        #     a=36

        logits = torch.squeeze(torch.matmul(output, w_t))

        # logits = torch.squeeze((output))


        probs = F.softmax(temperature * logits, dim=1)
        # m=0\
        # n = Categorical(probs)
        probnp=probs.cpu().detach().numpy()

        # test prob, make  variance more
        # if(is_gen==True):
        #     temp=probs*probs
        #     sumtemp=torch.sum(temp,dim=1).unsqueeze(1)
        #     # sumkk=sumtemp.unsqueeze(1)
        #
        #     probs=torch.div(temp,sumtemp)

        # test blank +make variance more

        # if(is_gen==True):
        #     probs[:,15]=probs[:,15]+probs[:,16]
        #     probs[:,16]=0
        #     tempkeep=probs[:,15]
        #     temp = probs * probs
        #     a=torch.sum(temp,dim=1)
        #     sumtemp=(torch.sum(temp,dim=1)-tempkeep*tempkeep).unsqueeze(1)
        #
        #     probs=torch.div(temp,sumtemp)
        #     probs[:,15]=tempkeep

        x_tp1 = Categorical(probs).sample()

        # self.keep_time_num=(is_keep.long())*(self.keep_time_num[:,:]+1) #use 7 represent the value more than 7
        # self.blank_time_num=(is_blank.long())*(self.blank_time_num[:,:]+1)

        return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1, \
               last_goal_temp, real_goal, sub_goal, probs, t + 1



#zhu---------------------









