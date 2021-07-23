import numpy as np
import json

with open("../params/train_params.json", 'r') as f:
    train_params = json.load(f)
# os.environ['CUDA_VISIBLE_DEVICES'] = train_params['gpu_number']

# device_ids=[0,1]


log=open("../trainlog.txt")
import torch
import torch.nn as nn
import torch.optim as optim

from utils import G_gpu,C_gpu
from main import get_arguments,prepare_model_dict
from critic import  Critic
import  data_iter



def cnm():
    use_cuda = torch.cuda.is_available()
    # Random seed
    param_dict = get_arguments()
    torch.manual_seed(param_dict["train_params"]["seed"])
    # Pretrain step
    # checkpoint_path = param_dict["train_params"]["checkpoint_path"]
    # if checkpoint_path is not None:
    #     checkpoint = restore_checkpoint(checkpoint_path)
    #     model_dict = checkpoint["model_dict"]
    #     optimizer_dict = checkpoint["optimizer_dict"]
    #     scheduler_dict = checkpoint["scheduler_dict"]
    #     ckpt_num = checkpoint["ckpt_num"]
    # else:
    model_dict = prepare_model_dict(use_cuda)



    lr_dict = param_dict["train_params"]["lr_dict"]
    # optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)

    params={
        "seq_len": 128,
        "num_classes": 2,
        "vocab_size": 32,
        "dis_emb_dim": 1024,
        "filter_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        "num_filters": [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
        "start_token": 0,
        "goal_out_size": None,
        "step_size": 16,
        "dropout_prob": 0.2,
        "l2_reg_lambda": 0.2
    }
    # discriminator = model_dict['discriminator']
    discriminator = Critic(**params)
    discriminator=discriminator.cuda(C_gpu)
    positive_filepath = "../data/train_corpus.npy"
    # negative_filepath="./data/gen_corpus.npy"
    # test_filepath = "./exp/obv-cnn-test.npy"
    # negative_filepath = "./exp/obv-cnn-train.npy"
    # test_filepath = "./exp/obv-sf-test.npy"
    test_filepath = "./exp/obv-cnn-test.npy"
    # negative_filepath = "./exp/obv-sf-train.npy"
    negative_filepath = "./exp/obv-cnn-train.npy"

    # positive_filepath= "./data/train_corpus.npy"
    # # negative_filepath="./data/gen_corpus.npy"
    # negative_filepath = "./data/gen_corpus.npy"
    # generate_samples(model_dict, negative_filepath, 156, use_cuda, 1)

    # model_dict['generator'].load_state_dict(torch.load("./pts/adv_gen10.pt"))
    model_dict['generator'] = model_dict['generator'].cuda(G_gpu)

    # generate_samples(model_dict,"obv selfatt test.npy", 20, use_cuda, 1)

    # generate_samples(model_dict, negative_filepath, 156, use_cuda, 1)

    print("xxxxxxxxxx number of paramerters in networks is {}  ".format(sum(x.numel() for x in discriminator.parameters())))
    # print("generated finished")
    lr=5e-5
    optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    enloss = torch.nn.CrossEntropyLoss()
    epochs=0
    if(epochs==1):
        # discriminator.load_state_dict(torch.load("ob-cnn-citic-cnn.pt"))
        discriminator.load_state_dict(torch.load("ob-cnn-citic-cnn+l2.pt"))
    epochs =500
    for epoch in range(epochs):
        dis_data_loader = data_iter.dis_narrow_data_loader(positive_filepath, negative_filepath, 64, True, 1, False)
        print_l2loss=0
        print_loss = 0
        for i,sample in enumerate(dis_data_loader):

            data, label = sample["data"], sample["label"]
            data=data.cuda(C_gpu)
            label=label.cuda(C_gpu)
            if (data.shape[0]==64):
                prob=discriminator(data)


                # l2loss=discriminator.l2_loss()
                # loss = enloss(prob["score"], label.view(-1))+l2loss
                l2_loss=discriminator.l2_loss()
                loss = enloss(prob["score"], label.view(-1))+l2_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print_loss+=loss.item()
                print_l2loss +=l2_loss.item()
                if(i%63==0):
                    print(print_loss/63,print_l2loss/63)
                    print_l2loss=0
                    print_loss=0
        torch.save(discriminator.state_dict(), "obv-cnn-critic-cnn+l2.pt")
        del dis_data_loader
        print (epoch)
    # negfile="./data/gen_absolute_corpus.npy"
    #     del dis_data_loader





    i=0
    # neg="ncc"
    # while i<=21:
    #     print(i)
    #     strrrr= "./pts/adv_gen"+str(i)+".pt"
    #     negative_filepath=neg+str(i)+".pt"
    #     model_dict['generator'].load_state_dict(torch.load(strrrr))
    #
    #
    #     i=i+1
    # testdata_filepath = "testdata.npy"

    # generate_samples(model_dict, testdata_filepath, 3, use_cuda, 1)






    dis_data_loader = data_iter.dis_narrow_data_loader(positive_filepath, test_filepath, 64, True, 4, False)
    acc=np.zeros([0,1],dtype=np.long)
    summm=0

    for i, sample in enumerate(dis_data_loader):
        # sample=sample.cuda()
        data, label = sample["data"], sample["label"]
        label=label.cuda(C_gpu)
        data=data.cuda(C_gpu)
        if (data.shape[0] == 64):
            prob = discriminator(data)
            prob_label=torch.argmax(prob['score'],dim=1)

            for j in range(64):
                if prob_label[j]==label[j]:
                    summm+=1
        else:
            print(summm/(i*64))
            break

    print(summm/((i+1)*64))
cnm()








