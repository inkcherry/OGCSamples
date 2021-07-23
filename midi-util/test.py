import  numpy as np
import  re
import os
from p2m import mmat_2_mmidi,t2mat_2_mmidi
import  interval2onehot
from interval2onehot import real_to_d4onehot,interval_to_real,d4onehot_to_real,add_pad_of_interval,print_is_same_of_real, real_to_interval,chord_to_number,number_to_chord
from shutil import copyfile
def fun(type):


    if type=="melody128_to_notonehot":
        t = np.load("melody128.npy")
        ta=d4onehot_to_real(t)
        # pp = d4onehot_to_real(za[3729:3730, :, :])
        np.save("train_absolute_corpus.npy",ta)
        a=3
    if type=="xx":
        a=np.zeros((1,1,128,128),dtype=np.float32)

        b=a[:,:,:,24:108]
        cc=3
    if type=="get_length_np":
        gg=np.load("music_length_count.npy")
        ci=2
    if type=="re":
        name="fdsafasdfkkkss"
        pattern="kkk"
        mm=re.search(pattern,name)
        return
    if type=="get_result":

        # for i in range(2):
        #     num=50+5*i
        # tempnpyname="npydir/eval_data2_17.npy"

        tempnpyname = "npydir/chord128.npy"


        tempnpy=np.load(tempnpyname)




        tempnpy = tempnpy.reshape((tempnpy.shape[0], tempnpy.shape[1], tempnpy.shape[2], 1))
        for i in range(tempnpy.shape[0]):
            j=1
            while(j<=8):
                tempnpy[i, j * 16-1, 12:20, 0] = 1
                j+=1
        # tempnpy=real_to_d4onehot(tempnpy)
        name,suffix=os.path.splitext(tempnpyname)
        save_name=name+".mid"
        mmat_2_mmidi(tempnpy, save_name, tempo=30)

    if type == "getinterval_countsafddsaf":
        pos,neg=interval2onehot.get_interval_count()
        c=0
        print("nmmm")
        return
    if type == "getinterval_result":
        name_list=[]

        # i=140
        i=0
        while i <=0 :
            # name_list.append("eval_dat0a0_"+str(i)+".npy")
            #
            name_list.append("stRL.npy")
            # name_list.append("eval_dataadv_0_" + str(i) + ".npy")
            # name_list.append("eval_data" + str(i) + "_0.npy")
            # name_list.append("eval_data" + str(i) + "_1.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_0.npy")
            # name_list.append("ncc"+str(i)+".pt.npy")
            # name_list.append("eval_dataadv_"+str(i)+"_1.npy")

            i+=1

        source_path = "/home/liumingzhi/inkcpro/20201104haode/data/evaldir"
        # source_path="./npydir"
        # source_path = "/home/liumingzhi/inkcpro/20200807x2/data/evaldir"

        target_path="npydir"
        for epoch in range(len(name_list)):
            name=name_list[epoch]
            source=os.path.join(source_path,name)
            target=os.path.join(target_path,name)
            # copyfile(source, target)

            tempnpyname = source
            tempnpy=np.load(tempnpyname)
            # tempnpy=tempnpy[0:200,]

            name, suffix = os.path.splitext(target)
            save_name = name + ".mid"
            tempnpy=interval_to_real(tempnpy)
            # np.save(tempnpy,name+"_absolute.npy")


            np.save(name+"absolute.npy",tempnpy)
            print("save absolute copy")
            for i in range(tempnpy.shape[0]):
                for j in range(tempnpy.shape[1]):
                    if tempnpy[i][j]>84 or tempnpy[i][j]<0:
                        print("sdf")
            d4 = real_to_d4onehot(tempnpy)
            mmat_2_mmidi(d4, save_name, tempo=30)

            print(tempnpyname+"    "+save_name)
            print(i)

    if type == "getabsolute_result":
        name_list = []
        # for i in range ():
        # name_list.append("eval_data0_" uuuuuuuuuu+ str(i) + ".npy")
        # name_list.append("eval_data0_" + str(i) + ".npy")
        # i=140
        i = 0
        while i <= 0:

            # name_list.append("obv-sf-absolute_pre.npy")
            # name_list.append("OK的absolute.npy")
            # name_list.append("trainhear500.npy")
            name_list.append("adv39.npy")
            i += 1



        name = ""

        # source_path = "/home/liumingzhi/inkcpro/20200919absolutegoG/data/evaldir"

        source_path = "./npydir"

        target_path = "npydir"
        for epoch in range(len(name_list)):
            name = name_list[epoch]
            source = os.path.join(source_path, name)
            target = os.path.join(target_path, name)
            # copyfile(source, target)

            tempnpyname = source
            tempnpy = np.load(tempnpyname)

            name, suffix = os.path.splitext(target)
            save_name = name + ".mid"
            # tempnpy = interval_to_real(tempnpy)
            # np.save(tempnpy,name+"_absolute.npy")

            np.save(name + "absolute.npy", tempnpy)
            tempnpy = tempnpy[0:200,:]
            print("save absolute copy")
            for i in range(tempnpy.shape[0]):
                for j in range(tempnpy.shape[1]):
                    if tempnpy[i][j] > 84 or tempnpy[i][j] < 0:
                        print("sdf")
            d4 = real_to_d4onehot(tempnpy)
            mmat_2_mmidi(d4, save_name, tempo=30)

            print(tempnpyname + "    " + save_name)
            print(i)

    if type == "test_length":
        name_list = []
        # for i in range ():
        # name_list.append("eval_data0_" + str(i) + ".npy")
        # name_list.append("eval_data0_" + str(i) + ".npy")
        # i=140
        i = 0
        while i <= 100:
            # name_list.append("eval_data0_"+str(i)+".npy")
            # name_list.append("eval_dataadv_" + str(0) + "_.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_1.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_2.npy")

            # name_list.append("eval_dataadv_" + str(i) + ".npy")
            # name_list.append("eval_dataadv_0_"+str(i)+".npy")

            # for j in range(3):
            #     name_list.append("eval_dataadv_" + str(i) + "_"+str(j)+".npy")
            # name_list.append("eval_dataadv_" + str(i) + "_1.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_2.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_3.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_4.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_5.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_6.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_7.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_8.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_9.npy")

            # name_list.append("eval_dataadv_" + str(i) + "_2.npy")
            # name_list.append("eval_dataadv_" + str(i) + "_3.npy")
            # name_list.append("gennnn_data.npy")

            # name_list.append("obv-cnn-test.npy")
            # name_list.append("obv-sf-test.npy")
            i=i+1

        name = ""
        # source_path="/home/liumingzhi/inkcpro/20200523i1/data/evaldir"
        # source_path="/home/liumingzhi/inkcpro/20200601i1/data/evaldir"
        # source_path = "/home/liumingzhi/inkcpro/20200601i1/data/evaldir"
        source_path = "/home/liumingzhi/inkcpro/20200807x2/data/evaldir"
        target_path = "npydir"
        for epoch in range(len(name_list)):
            name = name_list[epoch]
            source = os.path.join(source_path, name)
            target = os.path.join(target_path, name)
            copyfile(source, target)

            tempnpyname = source
            tempnpy = np.load(tempnpyname)

            name, suffix = os.path.splitext(target)
            save_name = name + ".mid"

            # interval   token
            # +1~+15    [0]~[14]
            # 0(keep)        [15]
            # null           [16]
            # -1~-15   [17]~[31]
            tempnpy[:,0:6]=15
            tempnpy[:,7]=16

            tempnpy[:, 8:15] = 15
            tempnpy[:, 16] = 16
            tempnpy[:, 17:32] = 15

            tempnpy = interval_to_real(tempnpy)

            np.save("absolute.npy", tempnpy)
            for i in range(tempnpy.shape[0]):
                for j in range(tempnpy.shape[1]):
                    if tempnpy[i][j] > 84 or tempnpy[i][j] < 0:
                        print("sdf")
            d4 = real_to_d4onehot(tempnpy)
            mmat_2_mmidi(d4, save_name, tempo=30)

            print(source_path + name)
            print(i)

    if type == "get_many":
        source_path = "/home/liumingzhi/inkcpro/20200807x2/data/evaldir/"
        melody_track_name = "eval_data74_1.npy"
        melody_track_file = source_path+melody_track_name
        print(melody_track_file)

        chord_track_file = "npydir/chord128.npy"

        chord_track = np.load(chord_track_file)
        melody_track = np.load(melody_track_file)
        # def merge_onehot(melody_track,chord_track):
        #     tempnpy = interval_to_real(melody_track)
        #     d4mt = real_to_d4onehot(tempnpy)
        #     d4ct_ = np.expand_dims(chord_track, 3)
        #
        #     d4ct=d4ct_[0:d4mt.shape[0],:,:,:]
        #     merge_track= d4ct+d4mt
        #     merge_track[merge_track==2]=1
        #     return merge_track
        #
        # #tag
        #
        #
        # merge_track=merge_onehot(melody_track,chord_track)
        #
        # merge_track[:,:,36,:]=1
        # mmat_2_mmidi(merge_track, "merge"+melody_track_name+".mid", tempo=30)
        # print("finished")
        tempnpy = interval_to_real(melody_track)
        d4mt = real_to_d4onehot(tempnpy)
        d4ct_ = np.expand_dims(chord_track, 3)
        d4ct = d4ct_[0:d4mt.shape[0], :, :, :]
        gg=np.concatenate((d4ct,d4mt),axis=3)

        # mmat_2_mmidi(tempnpy, save_name, tempo=30)
        t2mat_2_mmidi(gg,"nmsl.mid",tempo=180)
        print("sdfdfssss")

    if type== "add_pad":
        source_path="/home/liumingzhi/inkcpro/20201103addpad/data/train_corpus.npy"
        train_data= np.load(source_path)
        cc = train_data[0:100,:]

        kk=add_pad_of_interval(cc)



        a=3

    if type=="xiaoyangben":
        mm=np.load("xiaoyangben_melody128.npy")
        cc=d4onehot_to_real(mm)
        cc=0

    if type =="addpad":
        za = np.load("melody128addpad.npy")
        pp = d4onehot_to_real(za)   # OK
        # np.save("addpadreal.npy",pp)
        # interval= real_to_interval(pp)
        #
        # np.save("addpadinterval.npy",interval)
        # rereal= interval_to_real(interval)






        d4 = real_to_d4onehot(pp[3000:3100,:])
        # d4=d4onehot_to_real(ss)

        mmat_2_mmidi(d4, "woaini.mid", tempo=120)


        #
        # kk = ss[:,:,:,0]


        pp = 3
    if type == "shujuji":
        za = np.load("withpadmelody128.npy")
        pp = d4onehot_to_real(za)  # OK
        d4 = real_to_d4onehot(pp[3000:3100, :])

        np.save("melody_id.npy",pp)

        # d4=d4onehot_to_real(ss)

        mmat_2_mmidi(d4, "woaini.mid", tempo=150)
        print("shujujiwanle")
        a=3
    if type=="chord2number":
        chord_to_number("chord128final.npy")

    if type == "number2chord":
        number_to_chord("chord_id_track.npy", "name_chord.npy")

    if type == "get_melody_np":
        cc = np.load("melody128final.npy")
        print("jkl")
    else : print("no fun")
# fun("chord2number")
# fun("get_melody_np")
print("sdfdsf")
# fun("getabsolute_result")bb
fun("shujuji")