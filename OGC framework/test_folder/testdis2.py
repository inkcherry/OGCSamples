import numpy as np
import json
import os #for checkpoint management

with open("../params/train_params.json", 'r') as f:
    train_params = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = train_params['gpu_number']
# device_ids=[0,1]


log=open("../trainlog.txt")
import torch
import torch.nn as nn
import torch.optim as optim

from data_iter import dis_data_loader
from dis2 import Discriminator_Absolute
from main import get_arguments,prepare_model_dict,prepare_optimizer_dict
from torch.utils.data import Dataset, DataLoader


class Dataset():
    def __init__(self):
        pos_data = torch.zeros([992,128],dtype=torch.long)

        pos_data.fill_(5)
        neg_data = torch.zeros([992,128],dtype=torch.long)
        neg_data.fill_(10)

        np.random.shuffle(pos_data)
        data_number=min(pos_data.shape[0],neg_data.shape[0])

        pos_data=pos_data[0:data_number,:]
        neg_data = neg_data[0:data_number, :]
        #make len(pos)===len(neg)

        # print("Pos data: {}".format(len(pos_data)))
        # print("Neg data: {}".format(len(neg_data)))
        pos_label = np.array([1 for _ in pos_data])
        neg_label = np.array([0 for _ in neg_data])
        self.data = np.concatenate([pos_data, neg_data])
        self.label = np.concatenate([pos_label, neg_label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).long()
        label = torch.nn.init.constant_(torch.zeros(1), int(self.label[idx])).long()
        return {"data": data, "label": label}

def dis_data_loader(batch_size=32, shuffle=1, num_workers=0, pin_memory=False):
    dataset = Dataset()
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)




params={
"embedding_dim":64,
"hidden_dim":64,
"vocab_size":85,
"label_size":2,
"batch_size":64,
"seq_len":128,
"is_bidirectional":True
}

lr=0.00005

def fixed_data_test_dis():

    discriminator_absolute = Discriminator_Absolute(**params)

    dataloader=dis_data_loader()
    optimizer=optim.Adam(discriminator_absolute.parameters(), lr=lr)
    enloss=torch.nn.CrossEntropyLoss()
    for i,sample in enumerate(dataloader):
        data, label = sample["data"], sample["label"]
        prob=discriminator_absolute(data)
        loss=enloss(prob,label.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if(i%10==0):
            print(loss)

    k1 = torch.zeros([32, 128], dtype=torch.long)
    k1.fill_(5)
    k2 = torch.zeros([32, 128], dtype=torch.long)
    k2.fill_(10)

    l1=discriminator_absolute(k1)
    l2=discriminator_absolute(k2)
    b=3


import  data_iter
def test_dis():
    use_cuda = torch.cuda.is_available()
    # Random seed
    param_dict = get_arguments()
    torch.manual_seed(param_dict["train_params"]["seed"])
    # Pretrain step
    checkpoint_path = param_dict["train_params"]["checkpoint_path"]
    # if checkpoint_path is not None:
    #     checkpoint = restore_checkpoint(checkpoint_path)
    #     model_dict = checkpoint["model_dict"]
    #     optimizer_dict = checkpoint["optimizer_dict"]
    #     scheduler_dict = checkpoint["scheduler_dict"]
    #     ckpt_num = checkpoint["ckpt_num"]
    # else:
    model_dict = prepare_model_dict(use_cuda)
    lr_dict = param_dict["train_params"]["lr_dict"]
    optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)



    discriminator_absolute = Discriminator_Absolute(**params)
    discriminator_absolute=discriminator_absolute.cuda()
    negfilepre = "./data/gennnn_data.npy"
    # negfile="./data/genxxx_data.npy"
    negfile="./data/eval_data153_1.npy"
    posfile = "./data/train_absolute_corpus.npy"



    #generated samples
    # generate_samples(model_dict,negfilepre,353, True, 1)
    # rn=np.load(negfilepre)
    # rn2=interval_to_real(rn)
    # np.save(negfile,rn2)



    optimizer = optim.Adam(discriminator_absolute.parameters(), lr=lr)
    enloss = torch.nn.CrossEntropyLoss()
    for epoch in range(50):
        dis_data_loader = data_iter.dis_narrow_data_loader(posfile, negfile, 64, True, 4, False)
        for i,sample in enumerate(dis_data_loader):

            data, label = sample["data"], sample["label"]
            data=data.cuda()
            label=label.cuda()
            if (data.shape[0]==64):
                prob=discriminator_absolute(data)

                a=prob
                b=label.view(-1)
                loss=enloss(prob,label.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if(i%10==0):
                    print(loss)


    negfile="./data/gen_absolute_corpus.npy"

    dis_data_loader = data_iter.dis_narrow_data_loader(posfile, negfile, 64, True, 4, False)
    acc=np.zeros([0,1],dtype=np.long)
    sum=0
    for i, sample in enumerate(dis_data_loader):
        # sample=sample.cuda()
        data, label = sample["data"], sample["label"]
        data=data.cuda()
        label=label.cuda()
        if (data.shape[0] == 64):
            prob = discriminator_absolute(data)
            prob_label=torch.argmax(prob,dim=1)

            for j in range(64):
                if prob_label[j]==label[j]:
                    sum+=1
    print(sum/(i*64))

    # print(np.mean(acc))





test_dis()
# for i in range(31):







# pretrain_discriminator_absolute(0,"",model_dict=,optimizer_dict=,scheduler_dict=,dis_data_loader,vocab_size=,positive_file=,negative_file=,batch_size=
#         epochs, use_cuda =, temperature =)


