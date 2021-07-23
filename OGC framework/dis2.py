import math
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from scipy.stats import truncnorm
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Highway(nn.Module):
    #Highway Networks = Gating Function To Highway = y = xA^T + b
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(in_size, out_size)
    def forward(self, x):
        #highway = F.sigmoid(highway)*F.relu(highway) + (1. - transform)*pred # sets C = 1 - T
        g = F.relu(self.fc1)
        t = torch.sigmoid(self.fc2)
        out = g*t + (1. - t)*x
        return out
class Discriminator_Absolute_Cnn(nn.Module):
    """
    A CNN for text classification
    num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
    """
    def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim,
                    filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
        super(Discriminator_Absolute_Cnn, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.start_token = start_token
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)

        #Building up layers
        self.emb = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, self.dis_emb_dim)) for f_size, num_f in zip(self.filter_sizes, self.num_filters)
        ])
        for conv in self.convs:
            conv.bias.data.fill_(0.1)
        self.highway = nn.Linear(self.num_filters_total, self.num_filters_total)
        #in_features = out_features = sum of num_festures
        self.dropout = nn.Dropout(p = self.dropout_prob)
        #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
        #Each channel will be zeroed independently onn every forward call
        self.fc = nn.Linear(self.num_filters_total, self.num_classes)

    def forward(self, x):
        """
        Argument:
            x: shape(batch_size * self.seq_len)
               type(Variable containing torch.LongTensor)
        Return:
            pred: shape(batch_size * 2)
                  For each sequence in the mini batch, output the probability
                  of it belonging to positive sample and negative sample.
            feature: shape(batch_size * self.num_filters_total)
                     Corresponding to f_t in original paper
            score: shape(batch_size, self.num_classes)

        """
        #1. Embedding Layer
        #2. Convolution + maxpool layer for each filter size
        #3. Combine all the pooled features into a prediction
        #4. Add highway
        #5. Add dropout. This is when feature should be extracted
        #6. Final unnormalized scores and predictions
        # -1 for pad, all add +1 ||  max for pad

        # x=x.detach().cuda(1)
        emb = self.emb(x).unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
        pooled_out = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pooled_out, 1) # batch_size * sum(num_filters)
        #print("Pred size: {}".format(pred.size()))
        # highway = self.highway(pred)
        #print("highway size: {}".format(highway.size()))
        highway = torch.sigmoid(pred)* F.relu(pred) + (1.0 - torch.sigmoid(pred))*pred
        features = self.dropout(highway)
        # features=highway
        score = self.fc(features)
        score = torch.sigmoid(score)
        pred = F.log_softmax(score, dim=1) #batch * num_classes

        # featurenp=features.detach().cpu().numpy()
        # debugtag=[]
        # for i,v in enumerate([emb,convs,pooled_out,highway,featuresssstag,features]):
        #     j=v.detach()
        #     debugtag.append(j.cpu().numpy())
        # npemb,convs,pooled_out,highway,features=emb.detach().numpy(),convs.detach().numpy(),pooled_out.detach().numpy(),highway.detach().numpy()
        return {"pred":pred, "feature":features.detach(), "score": score}


    def l2_loss(self):
        W = self.fc.weight
        b = self.fc.bias
        l2_loss = (torch.sum(W*W) + torch.sum(b*b))/2
        l2_loss = self.l2_reg_lambda * l2_loss
        return l2_loss





#
# class Discriminator_Absolute_tn(nn.Module):
#
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,batch_size,seq_len,num_layers=1,is_bidirectional=False):
#         super(Discriminator_Absolute, self).__init__()
#         self.is_bidirectional=is_bidirectional
#         self.num_layers=num_layers
#         self.hidden_dim = hidden_dim
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=num_layers,bidirectional=is_bidirectional)
#         self.bid_flag = 1
#         if(is_bidirectional):
#             self.bid_flag=2
#
#         self.attention = nn.Linear(self.bid_flag * self.hidden_dim, 1)
#         self.fc = nn.Linear(hidden_dim*self.bid_flag, 2)
#
#         self.batch_size=batch_size
#         self.seq_len=seq_len
#         self.attention_dropout = nn.Dropout(p=0.5)
#
#     def init_hidden(self):
#         # the first is the hidden h
#         # the second is the cell  c
#
#         h = Variable(torch.zeros(self.num_layers*self.bid_flag,self.batch_size, self.hidden_dim))
#         c = Variable(torch.zeros(self.num_layers*self.bid_flag,self.batch_size, self.hidden_dim))
#
#         return h.cuda(),c.cuda()
#     def forward(self, sentence):  #0.97  loss=0.06
#
#         #sentence: batch x seq_len
#         #embeds: batch x seq_len X emb_dim
#         #x: seq_len x batch x emb_dim
#         #h,c : (num_layers*num_directions), batch, hidden_size
#         #unpermute_out :seq_len, batch, (num_directions * hidden_size)
#
#         #y : batch xlabel_size
#         #y_prob
#
#         embeds = self.word_embeddings(sentence)
#         x=embeds.permute(1, 0, 2)
#         h,c=self.init_hidden()
#         unpermute_out, (h_t, c_t) = self.lstm(x,(h,c))
#         out=unpermute_out.permute(1,0,2)
#         unnormalize_weight = F.tanh(torch.squeeze(self.attention(out), 2))
#         unnormalize_weight = F.softmax(unnormalize_weight, dim=1)
#         normalize_weight = torch.nn.functional.normalize(unnormalize_weight, p=1, dim=1)
#         normalize_weight = normalize_weight.view(normalize_weight.size(0), 1, -1)
#         weighted_sum = torch.squeeze(normalize_weight.bmm(out), 1)
#         output = self.fc(self.attention_dropout(weighted_sum))
#
#         output=torch.sigmoid(output)
#         return output



#
#
#
# # def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,batch_size,seq_len,num_layers=1,is_bidirectional=False):
#
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # self.hidden_dim = hidden_dim
        # self.projection = nn.Sequential(
        #     nn.Linear(hidden_dim, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 1)
        # )

        #
        self.bias=True
        self.activation = F.relu
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, self.bias)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, self.bias)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, self.bias)




    def forward(self, encoder_outputs):

        # batch_size = encoder_outputs.size(0)
        # # (B, L, H) -> (B , L, 1)
        # energy = self.projection(encoder_outputs)
        # weights = F.softmax(energy.squeeze(-1), dim=1)
        # # (B, L, H) * (B, L, 1) -> (B, H)
        # outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # return outputs

        #
        q, k, v = self.linear_q(encoder_outputs), self.linear_k(encoder_outputs), self.linear_v(encoder_outputs)
        q = self.activation(q)
        k = self.activation(k)
        v = self.activation(v)
        dk = q.size()[-1]
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)
        attention = F.softmax(scores, dim=-1)
        out=attention.matmul(v)
        return out
        #

# def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,batch_size,seq_len,num_layers=1,is_bidirectional=False):

class Discriminator_Absolute(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,vocab_size,label_size,batch_size,seq_len,num_layers=1,is_bidirectional=False):
        super().__init__()
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim*seq_len, 2) # use the initial one is hidden_dim


        self.length = seq_len
        # self.length =torch.tensor([seq_len],torch.int64)
    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)

    def forward(self, inputs):
        inputs=inputs.permute(1,0).detach()
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        # packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, self.length)
        out, hidden = self.lstm(embedded)
        # out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding = self.attention(out.transpose(0, 1)) #[B,L,H]
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1)) #
        # (B, 2)
        outputs=torch.sigmoid(outputs)
        # , attn_weights
        return outputs