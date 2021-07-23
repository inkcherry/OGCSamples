import torch
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import  G_gpu
# A truncated distribution has its domain (the x-values) restricted to a certain range of values. For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
# def truncated_normal(shape, lower=-0.2, upper=0.2):
#     size = 1
#     for dim in shape:
#         size *= dim
#     w_truncated = truncnorm.rvs(lower, upper, size=size)
#     w_truncated = torch.from_numpy(w_truncated).float()
#     w_truncated = w_truncated.view(shape)
#     return w_truncated
#
# class Highway(nn.Module):
#     #Highway Networks = Gating Function To Highway = y = xA^T + b
#     def __init__(self, in_size, out_size):
#         super(Highway, self).__init__()
#         self.fc1 = nn.Linear(in_size, out_size)
#         self.fc2 = nn.Linear(in_size, out_size)
#     def forward(self, x):
#         #highway = F.sigmoid(highway)*F.relu(highway) + (1. - transform)*pred # sets C = 1 - T
#         g = F.relu(self.fc1)
#         t = torch.sigmoid(self.fc2)
#         out = g*t + (1. - t)*x
#         return out
# class Discriminator_Absolute_Cnn(nn.Module):
#     """
#     A CNN for text classification
#     num_filters (int): This is the output dim for each convolutional layer, which is the number
#           of "filters" learned by that layer.
#     """
#     def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim,
#                     filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
#         super(Discriminator, self).__init__()
#         self.seq_len = seq_len
#         self.num_classes = num_classes
#         self.vocab_size = vocab_size
#         self.dis_emb_dim = dis_emb_dim
#         self.filter_sizes = filter_sizes
#         self.num_filters = num_filters
#         self.start_token = start_token
#         self.goal_out_size = goal_out_size
#         self.step_size = step_size
#         self.dropout_prob = dropout_prob
#         self.l2_reg_lambda = l2_reg_lambda
#         self.num_filters_total = sum(self.num_filters)
#
#         #Building up layers
#         self.emb = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)
#
#         self.convs = nn.ModuleList([
#             nn.Conv2d(1, num_f, (f_size, self.dis_emb_dim)) for f_size, num_f in zip(self.filter_sizes, self.num_filters)
#         ])
#         for conv in self.convs:
#             conv.bias.data.fill_(0.1)
#         self.highway = nn.Linear(self.num_filters_total, self.num_filters_total)
#         #in_features = out_features = sum of num_festures
#         self.dropout = nn.Dropout(p = self.dropout_prob)
#         #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
#         #Each channel will be zeroed independently onn every forward call
#         self.fc = nn.Linear(self.num_filters_total, self.num_classes)
#
#     def forward(self, x):
#         """
#         Argument:
#             x: shape(batch_size * self.seq_len)
#                type(Variable containing torch.LongTensor)
#         Return:
#             pred: shape(batch_size * 2)
#                   For each sequence in the mini batch, output the probability
#                   of it belonging to positive sample and negative sample.
#             feature: shape(batch_size * self.num_filters_total)
#                      Corresponding to f_t in original paper
#             score: shape(batch_size, self.num_classes)
#
#         """
#         #1. Embedding Layer
#         #2. Convolution + maxpool layer for each filter size
#         #3. Combine all the pooled features into a prediction
#         #4. Add highway
#         #5. Add dropout. This is when feature should be extracted
#         #6. Final unnormalized scores and predictions
#         # -1 for pad, all add +1 ||  max for pad
#
#         # x=x.detach().cuda(1)
#         emb = self.emb(x).unsqueeze(1)
#         convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
#         pooled_out = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
#         pred = torch.cat(pooled_out, 1) # batch_size * sum(num_filters)
#         #print("Pred size: {}".format(pred.size()))
#         # highway = self.highway(pred)
#         #print("highway size: {}".format(highway.size()))
#         highway = torch.sigmoid(pred)* F.relu(pred) + (1.0 - torch.sigmoid(pred))*pred
#         features = self.dropout(highway)
#         # features=highway
#         score = self.fc(features)
#         score = torch.sigmoid(score)
#         pred = F.log_softmax(score, dim=1) #batch * num_classes
#
#         # featurenp=features.detach().cpu().numpy()
#         # debugtag=[]
#         # for i,v in enumerate([emb,convs,pooled_out,highway,featuresssstag,features]):
#         #     j=v.detach()
#         #     debugtag.append(j.cpu().numpy())
#         # npemb,convs,pooled_out,highway,features=emb.detach().numpy(),convs.detach().numpy(),pooled_out.detach().numpy(),highway.detach().numpy()
#         return {"pred":pred, "feature":features.detach(), "score": score}
#
#
#     def l2_loss(self):
#         W = self.fc.weight
#         b = self.fc.bias
#         l2_loss = (torch.sum(W*W) + torch.sum(b*b))/2
#         l2_loss = self.l2_reg_lambda * l2_loss
#         return l2_loss

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
        self.linear_v = nn.Linear(hidden_dim, 36, self.bias)




    def forward(self, encoder_outputs,use_feature,index_for_feature):

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
        #
        # if use_feature:
        #     cn=q[:, index_for_feature, :].unsqueeze(1)
        #     dkk=math.sqrt(dk)
        #     score1=q[:,index_for_feature,:].unsqueeze(1).matmul(k.transpose(-2, -1))/dkk
        #     score2=q.matmul(k[:,index_for_feature,:].unsqueeze(1).transpose(-2, -1))/ dkk
        #     attention1= F.softmax(score1, dim=-1) #B x 1xL
        #     attention2 = F.softmax(score2, dim=-2)# B x Lx 1
        #
        #     # attentionnp=attention1.detach().cpu().numpy()[:,0,:]
        #
        #     attention=attention1[:,0,:]+attention2[:,:,0]
        #     # attentionnppp=attention.detach().cpu().numpy()
        #     return attention
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)

        attention = F.softmax(scores, dim=-1)

            #scores for train manager


        out=attention.matmul(v)
        return out       #out for get prob

class Discriminator(nn.Module):
    """
    A CNN for text classification
    num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
    """
    def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim,
                    filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
        super().__init__()
        self.step_size = step_size
        self.dropout_prob = dropout_prob
        self.start_token = start_token
        self.seq_len=seq_len
        self.l2_reg_lambda = l2_reg_lambda
        self.input_dim = vocab_size
        # self.embedding_dim = dis_emb_dim
        self.embedding_dim =dis_emb_dim
        # self.hidden_dim = hidden_dimdi
        # self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim)
        self.embedding = nn.Embedding(vocab_size+1, self.embedding_dim,padding_idx=32)
        self.hidden_dim = 1024
        self.head_num=8
        self.vocab_size=vocab_size
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # self.attention = SelfAttention(self.hidden_dim)

        self.fc= nn.Linear(self.head_num*self.seq_len*36, 2)
        # self.dfc=nn.Linear(self.hidden_dim*self.hidden_dim,16)
        # self.dfc = nn.Linear(8192, 16)
        # self.fc_feature = nn.Linear(self.hidden_dim * self.seq_len*self.head_num, 1024)  # use the initial one is hidden_dim
        # self.fc = nn.Linear(1024, 2)

        self.length = seq_len
        # self.length =torch.tensor([seq_len],torch.int64)
        self.attention_list=[]
        self.multi_head_selfattention=nn.ModuleList([SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim),
                                                    SelfAttention(self.hidden_dim)])
    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)

    def l2_loss(self):
            W = self.fc.weight
            b = self.fc.bias
            l2_loss = (torch.sum(W*W) + torch.sum(b*b))/2
            l2_loss = self.l2_reg_lambda * l2_loss
            return l2_loss
    def forward(self, inputs,use_feature=False,index_for_feature=0):
        inpus = inputs.detach()
        inputs = inputs.permute(1, 0)
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        # packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, self.length)
        # out, hidden = self.lstm(embedded)
        # out = nn.utils.rnn.pad_packed_sequence(out)[0]
        # out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)

        # out = self.attention_list[0](embedded.transpose(0, 1))
        head0 = self.multi_head_selfattention[0](embedded.transpose(0, 1),use_feature,index_for_feature) #[B,L,H]
        head1 = self.multi_head_selfattention[1](embedded.transpose(0, 1),use_feature,index_for_feature)
        head2 = self.multi_head_selfattention[2](embedded.transpose(0, 1),use_feature,index_for_feature)
        head3 = self.multi_head_selfattention[3](embedded.transpose(0, 1),use_feature,index_for_feature)
        head4 = self.multi_head_selfattention[4](embedded.transpose(0, 1),use_feature,index_for_feature)
        head5 = self.multi_head_selfattention[5](embedded.transpose(0, 1),use_feature,index_for_feature)
        head6 = self.multi_head_selfattention[6](embedded.transpose(0, 1),use_feature,index_for_feature)
        head7 = self.multi_head_selfattention[7](embedded.transpose(0, 1),use_feature,index_for_feature) #use feature :B x 1 x dim    / not use feature B x seqlen x dim



        embedding= torch.cat([head0,head1,head2,head3,head4,head5,head6,head7],dim=-1)
        #(B,L,H*heads)





        embedding=embedding.view(batch_size, -1)
        #(B,L*H*heads)

        if use_feature:

            return {"feature": embedding.detach().cuda(G_gpu)}




        outputs = self.fc(embedding)  #(B,1024)

        # outputs = self.fc(feature)



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   


        # (B, 2)
        score = torch.sigmoid(outputs)

        pred = F.log_softmax(score, dim=1)

        return {"pred": pred, "feature":None, "score": score}
        # return {"pred": pred, "feature": feature, "score": score}
        # , attn_weights

        # return {"score": score}

