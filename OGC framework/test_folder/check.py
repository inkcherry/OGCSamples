import  numpy as np
import  torch
import torch.nn.functional as F
import  torch.nn
from scipy.special import expit
def main(func):
    if func=="test_train_data":
        def fun():
            gg=np.load("../data/train_corpus.npy")
            ff=1
            print(gg.shape)
            return
        return fun()

    if func=="softmax":
        def fun():
            ti=np.array(([1,2,3],[4,5,6]),dtype="int64")
            tt=torch.from_numpy(ti)
            prob=F.softmax(ti,dim=0)
            print(prob)
            return ti
            print("pp")
        return fun()
    if func=="gen_corpus":
        def fun():
            gg=np.load("../data/gen_corpus.npy")
            ff=1
            print(gg.shape)

            p=np.max(gg,axis=0)
            SFDSF=p
            return
        return fun()
    if func=="creat_ones_train_data":
        def fun():
            pp = np.ones([10000, 20], dtype="int64")
            np.save("../data/train_corpus.npy", pp)
            allone = np.load("../data/train_corpus.npy")
            real=np.load("train_corpus_back.npy")
            print(allone.shape)
            print(real.shape)
            return
        return fun()
    if func=="set_tensor_value":
        def fun():
            ti=np.array(([1,2,3],[4,0,6]),dtype="int64")
            tt=torch.from_numpy(ti)
            tm = torch.from_numpy(ti)
            torch.zero_(tt)

            tb=torch.tensor(([1,2,3],[2,4,5]))
            print((tb==2).nonzero())
            # print(torch.is_nonzero(tt)) #all zero
            # print(torch.is_nonzero(tm)) #have zero
            # print(torch.is_nonzero(tb)) #all no zero
        return fun()

    if func=="CrossEntropyLoss":
        input1=torch.tensor([[0,1,0],[0,0,1]],dtype=torch.float32,requires_grad=True)
        input2 = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32,requires_grad=True)
        input3 = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.float32,requires_grad=True)
        target=torch.tensor([1,2],dtype=torch.long)

        input4 = torch.randn(3, 5, requires_grad=True)
        target2 = torch.empty(3, dtype=torch.long).random_(5)

        f=torch.nn.CrossEntropyLoss()
        output0=f(input4,target2)
        output1 = f(input1, target)
        output2 = f(input2, target)
        output3 = f(input3, target)
        a=3

    if func=="eval_data":
        def fun():
            gg = np.load("eval_data.npy")
            ff = 1
            print(gg.shape)
            return

        return fun()

    if func=="l2_loss":
        def fun():
            w = torch.tensor([1, 2], dtype=torch.long)
            print(torch.sum(w*w))
            return
        return fun()

    if func=="cosines":
        w1 = torch.tensor([0.1, 0.2], dtype=torch.float32)
        w2 = torch.tensor([0.5, 0.2], dtype=torch.float32)
        print(F.cosine_similarity(w1,w2,dim=0))

        w1 = torch.tensor([0.1, 0.2], dtype=torch.float32)
        w2 = torch.tensor([0.2, 0.2], dtype=torch.float32)
        print(F.cosine_similarity(w1, w2, dim=0))

        w1 = torch.tensor([0.1, 0.2], dtype=torch.float32)
        w2 = torch.tensor([0.1, 0.2], dtype=torch.float32)
        print(F.cosine_similarity(w1, w2, dim=0))

        w1 = torch.tensor([0, 0.3], dtype=torch.float32)
        w2 = torch.tensor([0.3, 0.5], dtype=torch.float32)
        print(F.cosine_similarity(w1, w2, dim=0))

        w1 = torch.tensor([1, 1], dtype=torch.float32)
        w2 = torch.tensor([-0.3, -0.5], dtype=torch.float32)
        print(F.cosine_similarity(w1, w2, dim=0))

        w1 = torch.tensor([0,0.5,-0.1,0.2], dtype=torch.float32)
        w2 = torch.tensor([0.1,0.5,-0.1,0.2], dtype=torch.float32)
        print(F.cosine_similarity(w1, w2, dim=0))

    if func=="constant":
        n = torch.FloatTensor(3, 3, 4).fill_(0.5)
        w1 = torch.tensor([1, 1], dtype=torch.float32)
        w1.fill_(0.8)
        a=2
    if func=="logwrite":
        log = open("../data/trainlog.txt", "w")
        str="xxx"
        str2="ppp"
        log.write(str)
        log.write(str2)

    if func== "list npy replace":
        a=np.array([[1,2,3,4,5],[1,2,3,4,5]])
        lista=[]
        lista.append(0)
        lista.append(4)
        a[:,lista]=10
        print(a)
    if func =="int2onehot":
        # import torch

        batch_size = 5
        nb_digits = 10
        # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
        y = torch.LongTensor(batch_size, 1).random_() % nb_digits
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(batch_size, nb_digits)

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        print(y)
        print(y_onehot)
    if func == "mul":
        A = torch.tensor([1, 0])
        B = torch.tensor([[1, 2, 3], [4, 5, 6]])
        A = A.unsqueeze(1)
        C = A * B


    if func == "gather":
        # A = torch.tensor([[1, 2], [3, 4]])
        # B = torch.tensor([[ 0], 0]])
        # tt=torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
        A= torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
        B = torch.tensor([[1],[2],[0]])
        C = torch.gather(A,1,B)
        print(C)
    if func == "log":
        B = torch.tensor([[0.9],[0.5],[0.3],[0.1],[0.01],[0.001]])


        c=torch.log(B)/3
        print(c)
        a=1

    if func =="testshape":
        B = torch.tensor([[0.9], [0.5], [0.3], [0.1], [0.01], [0.001]])
        print(B.shape[0])

    if func == "npshuffle":
        A = np.array([[1,2,3],[4,5,6],[7,8,9]])
        B= np.arange(24)
        D =np.array([[1,2,3],[4,5,6],[7,8,9]])
        D=np.transpose(D,(1,0))


        np.random.shuffle(B)
        np.random.shuffle(A)
        np.random.shuffle(D)
        D = np.transpose(D, (1, 0))

        print("sd")
    if func =="save":
        g="data/evaldir/2b.npy"
        # g='./data/evaldir/eval_dataadv_0_0.npy'
        g='/home/liumingzhi/inkcpro/20200828goG/data/evaldir/2b.npy'
        l=np.array([1,2])
        np.save(g,l)
    if func =="sort":

        def rescal(reward):
            batch_size=reward.shape[0]
            orderc = np.argsort(reward, axis=0)
            order= np.argsort(orderc, axis=0)

            #
            rank = batch_size - order

            rescaled_rewards = expit(16 * (0.5 - rank / batch_size))
            return  rescaled_rewards

        a = np.array([[11, 12, 13], [13, 12, 11]])
        b = np.array([[11, 18, 13, 19], [13, 12, 11, 14], [3, 4, 6, 2]])
        k1=rescal(a)
        k2=rescal(b)
        # rescaled_rewards = np.transpose(rescaled_rewards)
        # return Variable(torch.from_numpy(rescaled_rewards)).float()
        return 0
    if func =="findindex":
        a=torch.tensor([[2,3,6],[3,4,5],[2,4,5]])
        print((a!=3)*(a!=4))

    if func == "get_intrinsic_rewards":

        def get_intrinsic_rewards(rewards, seq_len=128, step_size=16,vocal_size=32,batch_size=64):
            intrinsic_rewards = torch.zeros([batch_size,seq_len])
            total_intrinsic_rewards = torch.zeros([batch_size, seq_len])
            sub_intrinic_rewards = torch.zeros([batch_size, seq_len])

            steps =(int)(seq_len / step_size)
            final_reward = rewards[:, (steps-1)]




            Attenuation_coefficient_arry =torch.zeros([batch_size,seq_len])
            Attenuation_coefficient_arry[0:batch_size,:]=torch.arange(seq_len)

            sub_aca_exp=Attenuation_coefficient_arry%step_size+1
            total_aca_exp = Attenuation_coefficient_arry+1

            total_aca_standard =torch.zeros([batch_size,seq_len]).fill_(0.9)
            sub_aca_standard = torch.zeros([batch_size,seq_len]).fill_(0.8)


            total_aca=torch.pow(total_aca_standard,total_aca_exp)

            sub_aca=torch.pow(sub_aca_standard,sub_aca_exp)


            total_intrinsic_rewards=torch.unsqueeze(final_reward,1)*total_aca

            for i in range(steps):
                cur_rewards=torch.unsqueeze(rewards[:, i],1)
                sub_intrinic_rewards[:,i * step_size:  i * step_size+step_size] =sub_aca[:,i * step_size:  i * step_size+ step_size]* cur_rewards

            cc=(sub_intrinic_rewards+total_intrinsic_rewards).numpy()
            return sub_intrinic_rewards+total_intrinsic_rewards


        rewardss = torch.zeros([64,8]).fill_(1000)
        rewardss[:,2]=10
        get_intrinsic_rewards(rewardss)
        return 0

            # pn = get_intrinsic_rewards()
            #
            # end_reward = rewards[:, steps - 1]
            #
            # for i in range(steps):
    if func == "divide":
        ng= np.load("observer cnn 300pretrain.npy")
        tetst=ng[0:1000,:]
        train=ng[1001:11000,:]
        np.save("observer cnn test.npy",tetst)
        np.save("observer cnn train.npy", train)
        a=0

    if func =="view":
        a=np.load("../exp/obv-cnn-train.npy")
        b = np.load("../exp/obv-sf-train.npy")
        c = np.load("../exp/obv-cnn-test.npy")
        d = np.load("../exp/obv-sf-test.npy")
        fff=0
    if func =="torchview":
        B = torch.tensor([[1, 2, 3], [4, 5, 6]])
        C=B.view(3,-1)
        a=0
        return
    if func == "shanggucnn":
        # cc=np.load("../exp/shanggucnn.npy")

        # kn=cc[0:192,:]
        # knn = cc[192:, :]
        # cpp=np.load("../exp/shuijiao.npy")
        # wm =cpp[0:300,:]
        # np.save("../exp/shuijiaoxiao.npy",wm)
        # 
        # 
        # np.save("../exp/shanggutest.npy",kn)
        # np.save("../exp/shanggutrain.npy",knn)

        train = np.load("../data/train_corpus.npy")
        train_sm = train[0:500, :]
        np.save("../exp/trainhear.npy",train_sm)

        # ob_cnn=np.load("../exp/obv-cnn-test.npy")

        a=0

    
    else:
        print("No fun")


main("shanggucnn")


def print1(str):
    print(str)
    # log.write(str+"\n")

    # log = open("trainlog.txt", "w")
    # log.write("strr")

    log = open("../data/trainlog.txt", "w")

    log.write(str)

    return
# print1("cacaca")
# print(gg.shape)
#
# pp=np.ones([10000,20],dtype="int64")
# print(pp.shape)
# np.save("pp.npy",pp)

