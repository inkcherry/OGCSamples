import  numpy as np
import  tensorflow as tf
from tensorflow.keras.utils import to_categorical
from p2m import  p_2_mmat,mmat_2_mmidi
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
                elif(onehotmat[i][j][0]==255):   #pad token
                    real_data[i][j]=85
                # patch_blank_onehot[i][j][0]=1
    #to midi just cut this
    # np.concatenate(onehotmat,)
    return real_data





def add_pad_of_interval(interval_mat,rest_token=16):
    nums = interval_mat.shape[0]
    length=interval_mat.shape[1]
    for i in range(nums):
        for j in range(length):
            index=length-j-1
            if interval_mat[i][index]==16:
                interval_mat[i][index]=33    #pad ==33
            else:
                break

    return interval_mat

# def del_pad_of_interval():
#     return


def real_to_d4onehot(real_mat):   #in this function, all pad will be back to rest. just for play,  if we use d4onehot to reget real_mat, all pad_token will be replaced by rest.
    # real_mat=real_mat/100
    pad_token=85


    real_mat2=real_mat-(real_mat==pad_token) #reset 85 to 84 , pad to rest   delete 255成分
    d3onehot = to_categorical(real_mat2,85)




    #change to d4
    d4onehot=np.expand_dims(d3onehot[:,:,0:84],axis=3)
    return d4onehot


def real_to_d3onehot(real_mat):

    d3onehot = tf.keras.utils.to_categorical(real_mat,85)
    #change to d4
    # d4onehot=np.expand_dims(d3onehot[:,:,0:84],axis=3)
    return d3onehot

def tes_t():
    # ------------test code -------------------
    m = np.load("10midi.npy")  # one hot use 100
    # print(m)

    f = d4onehot_to_real(m)
    d4 = real_to_d4onehot(f)
    d4_100 = d4 * 100
    print(type(d4))
    exit()
    # g=(d4==m)
    # for i in range (g.shape[0]):
    #     for j in range(g.shape[1]):
    #         for k in range(g.shape[2]):
    #             for l in range(g.shape[3]):
    #                 if(g[i][j][k][l]==False):
    #                     print (i,j,k,l)
    #                     print(d4[i][j][k][l],m[i][j][k][l])
    # 0 0 0 0
    # 0 1 0 0



def print_is_same(d4_100,m):


    if ((d4_100 - m*100).any()):
        print("not same")
    else:
        print("same")

    # mmat_2_mmidi(d4, "de.mid")
    # mmat_2_mmidi(m, "m.mid")
    # mmat_2_mmidi(d4_100, "d4_100.mid")

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


def get_train_data():
    d4=np.load("new128length.npy")
    trainnpy=d4onehot_to_real(d4)
    np.save("realtrain.npy",trainnpy)
    judge=real_to_d4onehot(trainnpy)
    print_is_same(d4,judge)
    np.save("traindata.npy",d4)


def intervalvalue2id(interval_value):
    #clip
    if (interval_value < -15):
        interval_value = -15
    elif (interval_value > 15):
        interval_value = 15
    # interval   token
    # +1~+15    [0]~[14]
    # 0(keep)        [15]
    # null/blank          [16]
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
    pad = 32
    length=realnpy.shape[1]
    interval_array=np.zeros([realnpy.shape[0],length])
    # +1~+15    [0]~[14]
    # 0(keep)        [15]
    # null (84)          [16]
    # -1~-15   [17]~[31]
    #pad (85)   [32]

    for i in range(realnpy.shape[0]):
        templast=start_basic_pitch
        j=0
    #get start pitch
    #next pitch
    #from j to length -1
        while(j<=length-1):
            if(realnpy[i][j]!=84 and realnpy[i][j]!=85) : #if don't use pad ,delete this 85
                cur_interval=realnpy[i][j]-templast
                # cur_interval=__clip__(cur_interval)
                interval_array[i][j]=intervalvalue2id(cur_interval)
                templast=realnpy[i][j]
            elif realnpy[i][j]==84:  #null pitch
                interval_array[i][j]=null_pitch
            elif realnpy[i][j]==85:               #if don't use pad ,delete this line
                interval_array[i][j]=pad
            else:
                print("error in real2interval")
            j+=1
    return interval_array


def interval_to_real(intervalnpy):
    length=intervalnpy.shape[1]
    real_array=np.zeros([intervalnpy.shape[0],length])
    start_basic_pitch = 36
    null_pitch = 16
    pad=32
    p=0
    for i in range(intervalnpy.shape[0]):
        temp_last=start_basic_pitch
        j=0
        while(j<=length-1):
            if(intervalnpy[i][j]!=null_pitch and intervalnpy[i][j]!=pad):
                interval_value=id2interval(intervalnpy[i][j])
                real_array[p][j]=temp_last+interval_value
                temp_last = real_array[p][j]
                if(temp_last<0 or temp_last>=85): #throw the line
                    p=p-1  #index-1 but j+1  re value this line.
                    break

            elif (intervalnpy[i][j]==16):
                real_array[p][j]=84
            elif(intervalnpy[i][j]==32):                #if don't use interval delete this line
                real_array[p][j] = 85
            else:
                print("error in interval_to_real")
            j+=1
        p+=1
    return real_array[0:p,:]


def count_interval(realnpy):
    pos_interval_count=np.zeros([85],dtype=np.int)   #0~+85
    neg_interval_count=np.zeros([85],dtype=np.int)   #-1~-85
    start_pitch_count=np.zeros([85],dtype=np.int)
    templast=0
    length = realnpy.shape[1]
    for i in range(realnpy.shape[0]):
        templast=0
        j=0
    #get start pitch
        while(j<=length-1):
            #tag
            gg=realnpy[i][j]

            if(realnpy[i][j]!=84):            #not null
                start_pitch_count[realnpy[i][j]] += 1  # first pitch
                templast=realnpy[i][j]
                j+=1                              #next pitch
                break
            j+=1                              #next pitch
    #from j to length -1
        while(j<=length-1):
            if(realnpy[i][j]!=84):
                cur_interval=realnpy[i][j]-templast
                templast=realnpy[i][j]
                if(cur_interval>=0):
                    pos_interval_count[cur_interval]+=1
                else:
                    neg_interval_count[(-1)*cur_interval] += 1
            j+=1
    np.save("posintervalcount.npy",pos_interval_count,)
    np.save("negintervalcount.npy",neg_interval_count,)
    np.save("startpitchcount",start_pitch_count)

def get_interval_count():
    pos = np.load("posintervalcount.npy")
    neg = np.load("negintervalcount.npy")
    return pos,neg
def test_train_data():
    traindata=np.load("traindata.npy")
    len=traindata.shape[0]
    epoch=30
    offset=len//300

    for i in range(epoch):
        tempnpy = traindata[30000:30100, :, :, :]
        # tempnpy=traindata[offset*i:(i+1)*offset,:,:,:]
        mmat_2_mmidi(tempnpy, "split"+str(i)+".mid",tempo=30)

    tempnpy = traindata[offset * epoch, len-1, :, :]
    mmat_2_mmidi(tempnpy, "split" + str(epoch+1) + ".mid",tempo=30)

def testreal2interval2real():
    real = np.load("128bar4.npy")
    clip_real = real[180:190, :]
    re1real = np.load("rerereal.npy")
    clip_re1real=re1real[180:190,:]

    # intervalarr = real_to_interval(clip_real)
    # rerereal = interval_to_real(intervalarr)

    # mmat_2_mmidi(real_to_d4onehot(rerereal), "rerereal.mid",tempo=30)
    mmat_2_mmidi(real_to_d4onehot(clip_re1real), "clip_re1eal.mid",tempo=30)

    mmat_2_mmidi(real_to_d4onehot(clip_real), "clip_real.mid",tempo=30)


# testreal2interval2real()
def get_bar4train_data():
    real=np.load("128bar4.npy")
    intervalarr = real_to_interval(real)
    rerereal = interval_to_real(intervalarr)

    np.save("intervalnpy.npy",intervalarr)
    np.save("rerereal.npy",rerereal)


def judge_interva2real2interval_proc():
    kd=np.load("128bar4.npy")
    re1real=np.load("rerereal.npy")
    #--------
    # za=np.load("128bar4.npy")
    # ab=za[3729:3730,:]
    # kk=real_to_interval(ab)
    # mm=interval_to_real(kk)
    #
    # a=re1real[3729:3730,:]
    # b=real_to_interval(a)
    # c=interval_to_real(b)
    #____

    re2real=interval_to_real(real_to_interval(re1real))
    print_is_same_of_real(re1real,re2real)

def get_real_train_data_from_onehot_from_prco():
    za=np.load("melody128addpad.npy")
    pp=d4onehot_to_real(za[0:100,:,:])
    pp=3


def chord_to_number(file_chord):
    def is_named_chord(named_chord, named_count, cur_chord):
        chord_size = cur_chord.shape[0]
        for i in range(named_count):
            j = 0
            not_match=0
            while j < chord_size:
                if (named_chord[i][j] == cur_chord[j]):
                    j += 1
                    continue

                else:
                    j += 1
                    not_match=1
                    break
            while j < max_chord_size and not_match==0:
                if (named_chord[i][j] != 0):
                    not_match=1
                    break

                j += 1
            if(not_match==0):
                return i  #match number
        return 0  #no match ,new number
    def add_new_chord(named_chord, named_count, cur_chord):
        for i in range( min(cur_chord.shape[0],5)):
            named_chord[named_count,i]=cur_chord[i]

    chord_file=np.load(file_chord)
    max_chord_size=5
    named_chord=np.zeros([2000,max_chord_size],dtype=np.int64)
    named_chord.fill(0)

    named_count=1   #空和弦

    chord_file= chord_file[:,:,:]
    l=chord_file.shape[0]
    length=chord_file.shape[1]
    last=np.array([[0],[0],[0],[0]])

    chord_id_track=np.zeros([chord_file.shape[0],128],dtype=np.int64)
    chord_id_track.fill(0)

    for i in range(l):
        for j in range(length):

            if(np.max(chord_file[i][j])==0 or np.max(chord_file[i][j]) ==255):   #空音符
                chord_id_track[i][j]=0
                continue

            b = np.argwhere((chord_file[i][j] ==100))
            if (b.shape[0] == last.shape[0]):  # 等于上一个
                if ((b == last).all()):
                    chord_id_track[i][j]=last_chord_name
                    continue
            else:           # 由于和弦末尾导致的，缺失
                leak_but_same=True
                for mm in range(b.shape[0]):
                    if not (b[mm] in last):
                        leak_but_same=False
                        break
                if(leak_but_same):
                    chord_id_track[i][j] = last_chord_name
                    continue

            # b = np.argwhere((chord_file[i][j]>1) & (chord_file[i][j]!=254))
            # if(b[0]!=24):
            #     kkk=3
            name_id = is_named_chord(named_chord,named_count,b)
            last = b

            if name_id ==0:   #named_chord
                add_new_chord(named_chord, named_count, b)
                name_id=named_count
                named_count+=1

            last_chord_name = name_id
            chord_id_track[i][j]=name_id

        if (i % 100 == 0):
            print(i / 100,named_count)
    print(named_count)

    np.save("chord_id_track.npy",chord_id_track)
    np.save("name_chord.npy",named_chord)
    return
            # last_one = b


def number_to_chord(chord_id_file,named_chord_file):

    blank_id=0
    chord_id_track       = np.load(chord_id_file)
    name_chord           = np.load(named_chord_file)
    chord_track =  np.zeros([chord_id_track.shape[0],128,84])

    chord_id_track=chord_id_track[0:10,:]

    for i in range(chord_id_track.shape[0]):
        for j in range(chord_id_track.shape[1]):
            if(chord_id_track[i][j]==0):
                continue
            else:
                cur_chord_index=chord_id_track[i][j]
                b =  name_chord[cur_chord_index]
                print("sdfsdf")
                for  m in range(b.shape[0]):
                    if b[m] >0:          #-10 is pad
                        chord_track[i][j][b[m]] = 100

    real_chord_track = np.load("chord128final.npy")[0:10,:,:]
    return





# get_real_train_data_from_onehot_from_prco()
#get-------------train_data-------

# get_train_data()
# get_real_train_data_from_onehot_from_prco()
#
# get_real_train_data_from_onehot_from_prco()
#

#--------------get dataset-------------#
# get_bar4train_data()
# testreal2interval2real()
# get_real_train_data_from_onehot_from_prco()
# juduge_interva2real2interval_proc()
# get_bar4train_data()
# get_bar4train_data()
# c=np.load("intervalnpy.npy")
# m=3
# #