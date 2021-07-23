import numpy as np
import json
import os #for checkpoint management

with open("../params/train_params.json", 'r') as f:
    train_params = json.load(f)
# os.environ['CUDA_VISIBLE_DEVICES'] = train_params['gpu_number']
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# device_ids=[0,1]


log=open("../trainlog.txt")
import torch

from utils import get_rewards
from main import get_arguments,prepare_model_dict

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




import  data_iter
def testreward():
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




    discriminator_absolute = model_dict["discriminator_absolute"]
    # discriminator_absolute=discriminator_absolute.cuda()
    negfilepre = "./data/gen_corpus.npy"
    # negfile="./data/genxxx_data.npy"
    negfile="./data/absolute.npy"
    posfile = "./data/train_corpus.npy"
    #
    # positive_filepath =
    # negative_filepath = "./data/gennnn_data.npy"

    #generated samples
    # generate_samples(model_dict,negfilepre,50, True, 1)
    # rn=np.load(negfilepre)
    # rn2=interval_to_real(rn)
    # np.save(negfile,rn2)


    print("generated finished")

    epoch=1
    print("----pos reward test")
    for _ in range(epoch):
        dis_data_loader = data_iter.dis_narrow_data_loader(posfile, negfilepre, 64, True, 4, False)
        for i, sample in enumerate(dis_data_loader):
            data, label = sample["data"], sample["label"]
            data = data.cuda()
            label = label.cuda()
            if (data.shape[0] == 64):

                reward=get_rewards(model_dict,data,1,use_cuda=True)
                reward_np=reward.detach().cpu().numpy()
                label_np=label.detach().cpu().numpy()
                neg_reward_sum =0
                neg_count=0
                pos_reward_sum=np.zeros([8],dtype=np.float32)
                pos_count=0
                for i in range(64):
                    print(np.mean(reward_np[i],axis=0) ,label_np[i])
                    if (label_np[i]==0):
                        neg_reward_sum+=reward_np[i]
                        neg_count+=1
                    if (label_np[i]==1):
                        pos_reward_sum+=reward_np[i]
                        pos_count+=1

                for j in range(8):
                    print((neg_reward_sum[j]/neg_count),(pos_reward_sum[j]/neg_count))
                # print(neg_reward_sum/neg_count,pos_reward_sum/pos_count)
                # print()

                # prob = discriminator_absolute(data)
                #
                # a = prob
                # b = label.view(-1)
                # loss = enloss(prob, label.view(-1))
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()

    exit()

testreward()