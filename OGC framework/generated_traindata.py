from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
from test_folder.target import  Target,generated_train_data, nll_loss
import torch


class Fake_Dataset(Dataset):

    def __init__(self):
        with open("./params/target_params.json", 'r') as f:
            params = json.load(f)
        vocab_size=params['vocab_size']
        self.data = np.random.randint(vocab_size, size=(128, 20))

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()

def prepare_fake_data():
    dataset = Fake_Dataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=4)
    return dataloader


def run_target(use_cuda=False):
    with open("./params/target_params.json", 'r') as f:
        params = json.load(f)
    f.close()
    params['is_load']=0
    net = Target(**params)
    torch.save(net.state_dict(),params['net_paras_save_path'])  #save the target net
    if use_cuda:
        net = net.cuda()
    # dataloader = prepare_fake_data()
    # optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    # for i, sample in enumerate(dataloader):
    #     sample = Variable(sample)
    #     if use_cuda:
    #         sample = sample.cuda()
    #     optimizer.zero_grad()
    #     loss = target.loss_func(net, sample, use_cuda)
    #     loss.backward()
    #     optimizer.step()
    #
    #
    #     if i > 0:
    #         break
    train_data=generated_train_data(net,157)
    train_data_np=train_data.numpy()
    np.save("./data/train_corpus.npy",train_data_np)
    # gen = generate(net, use_cuda)
    # print(gen.size())
    print("Target test finished!")
    print("\n")

# t=1

def nll_loss(use_cuda=False):
    with open("./params/target_params.json", 'r') as f:
        params = json.load(f)
    f.close()
    net = Target(**params)
    dataloader = prepare_fake_data()
    for i, sample in enumerate(dataloader):
        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda()
        # optimizer.zero_grad()
        loss = nll_loss(net, sample, use_cuda)
        print("loss: ",loss)
        # loss.backward()
        # optimizer.step()



def main(func):
    if func=="generated_train_data":
        def fun():
            run_target(use_cuda=False)
            return
        return fun()
    elif func=="nll_loss":
        def fun():
            nll_loss(use_cuda=False)
            return
        return fun()
    else:
        print("no fun")




main("generated_train_data")

# print("sdfdsf")