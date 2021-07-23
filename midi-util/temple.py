import  numpy as np
import  tensorflow as tf
from p2m import  p_2_mmat,mmat_2_mmidi
from interval2onehot import  d4onehot_to_real,real_to_d4onehot
from file_batch import beat_resolution,is_monophonic

def d4_to_mid_save(npyfile ,savemidifile):
    re = np.load(npyfile)
    print(re.shape)
    mmat_2_mmidi(re, savemidifile, beat_reselution=beat_resolution)

def d4_to_real_save(d4,real):
    re = np.load(d4)
    print(re.shape)
    real_one = d4onehot_to_real(re)
    np.save(real, real_one)

def real_mid_save(real,midi):
    real_one = np.load(real)
    real4 = real_to_d4onehot(real_one)
    mmat_2_mmidi(real4, midi, beat_reselution=beat_resolution)


print("loading")
kk=np.load("allnokeyrealone.npy")

mnp=real_to_d4onehot(kk)

print(mnp.shape)
len=mnp.shape[0]
epoch=10
offset=(int)(len/epoch)



#太大，拆分成10份
i=0
while(i<epoch):
    filename="splitrealnokey"+str(i)+".mid"
    if(i==epoch-1):
        print(i*offset,len,filename)
        mmat_2_mmidi(mnp[i*offset:len,:,:,:], filename, beat_reselution=beat_resolution)
        i+=1
    else:
        print(i*offset,(i+1)*offset,filename)
        # kk=mnp[i*offset:(i+1)*offset,:,:,:]
        mmat_2_mmidi(mnp[i*offset:(i+1)*offset,:,:,:],filename,beat_reselution=beat_resolution)
        i+=1


# print(offset)




