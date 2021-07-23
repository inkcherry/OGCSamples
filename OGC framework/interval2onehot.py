import  numpy as np

np.set_printoptions(threshold=np.inf)
value_for_blank_not=84   #use 84 to represent blank pitch
def d4onehot_to_real(onehotmat): #although d3onehot_to_real
    onehotmat=onehotmat.reshape((onehotmat.shape[0],onehotmat.shape[1],onehotmat.shape[2]))
    # patch_blank_onehot=np.zeros(onehotmat.shape[0],onehotmat.shape[1],1)
    #use 85 to represent blank note
    real_data = np.argmax(onehotmat, axis=2)
    print(real_data.shape)
    for i in range(real_data.shape[0]):
        for j in range(real_data.shape[1]):
            if real_data[i][j]==0:  #是否argmax取得是默认的 即one-hot全0
                if(onehotmat[i][j][0]==0):    #argmax取默认的
                    real_data[i][j]=84   #84 represent null pitch
                # patch_blank_onehot[i][j][0]=1




    return real_data





def print_is_same(d4_100,m):


    if ((d4_100 - m*100).any()):
        print("not same")
    else:
        print("same")


def print_is_same_of_real(real1,rerereal):
    dd=real1-rerereal

    for i in range(dd.shape[0]):
        for j in range(dd.shape[1]):
            if dd[i][j]!=0:
                a=33


    if ((real1-rerereal).any()):
        print("not same")
    else:
        print("same")





def intervalvalue2id(interval_value):
    #clip
    if (interval_value < -15):
        interval_value = -15
    elif (interval_value > 15):
        interval_value = 15
    # interval   token
    # +1~+15    [0]~[14]
    # 0(keep)        [15]
    # null           [16]
    # -1~-15   [17]~[31]
    if(interval_value>0):
        return interval_value-1
    elif(interval_value<0):
        return (interval_value*-1)+16
    else:
        return 15

def id2interval(id):
    interval=0
    if id==15:
        interval=0
    elif id>=0 and id<=14:
        interval=id+1
    elif id>=17 and id<=31:
        interval=(id-16)*(-1)
    else:
        print("id2interval error")
    return interval

def real_to_interval(realnpy):
    start_basic_pitch = 36
    null_pitch=16

    length=realnpy.shape[1]
    interval_array=np.zeros([realnpy.shape[0],length])
    # +1~+15    [0]~[14]
    # 0(keep)        [15]
    # null           [16]
    # -1~-15   [17]~[31]

    for i in range(realnpy.shape[0]):
        templast=start_basic_pitch
        j=0
    #get start pitch
    #next pitch
    #from j to length -1
        while(j<=length-1):
            if(realnpy[i][j]!=84):
                cur_interval=realnpy[i][j]-templast
                # cur_interval=__clip__(cur_interval)
                interval_array[i][j]=intervalvalue2id(cur_interval)
                templast=realnpy[i][j]
            else:  #null pitch
                interval_array[i][j]=null_pitch

            j+=1
    return interval_array


def interval_to_real(intervalnpy,keep_size=False):
    if (keep_size==False):
        length=intervalnpy.shape[1]
        real_array=np.zeros([intervalnpy.shape[0],length])
        start_basic_pitch = 36
        null_pitch = 16
        p=0
        for i in range(intervalnpy.shape[0]):
            temp_last=start_basic_pitch
            j=0
            while(j<=length-1):
                if(intervalnpy[i][j]!=null_pitch):
                    interval_value=id2interval(intervalnpy[i][j])
                    real_array[p][j]=temp_last+interval_value
                    temp_last = real_array[p][j]
                    if(temp_last<0 or temp_last>=84): #throw the line
                        p=p-1
                        break

                else:
                    real_array[p][j]=84
                j+=1
            p+=1
        return real_array[0:p,:]
    else:
        overflow_line_list=[]
        length = intervalnpy.shape[1]
        real_array = np.zeros([intervalnpy.shape[0], length],dtype=np.int64)
        start_basic_pitch = 36
        null_pitch = 16
        p = 0
        for i in range(intervalnpy.shape[0]):
            temp_last = start_basic_pitch
            j = 0
            while (j <= length - 1):
                if (intervalnpy[i][j] != null_pitch):
                    interval_value = id2interval(intervalnpy[i][j])
                    real_array[p][j] = temp_last + interval_value
                    temp_last = real_array[p][j]
                    if (temp_last < 0 or temp_last >= 84):  # throw the line
                        real_array[p][j]=0  #make 1 2 ... 80 85 ...--->1 2 ... 80 0 0 0 0 ...
                        overflow_line_list.append(p) # record this line
                        break
                else:
                    real_array[p][j] = 84
                j += 1
            p += 1
        return real_array[0:p, :],overflow_line_list









# get_bar4train_data()
# testreal2interval2real()
# get_real_train_data_from_onehot_from_prco()
# juduge_interva2real2interval_proc()
# get_bar4train_data()
# get_bar4train_data()
# c=np.load("intervalnpy.npy")
# m=3