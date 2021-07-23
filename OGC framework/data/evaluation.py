
import  numpy as np

def is_leap(semitone):
    if (abs(semitone)>=5):
        if semitone>0:
            return 1,1 #is_leap, leap_direction
        return 1,-1
    return 0,0


# def law_of_recovery():



def law_of_recovery_information(realnpy):
    start_basic_pitch = 36
    null_pitch = 16
    length = realnpy.shape[1]
    interval_array = np.zeros([realnpy.shape[0], length])
    is_leap_flag=0
    direction=0
    law_of_recovery_count=0
    leap_count=0
    # +1~+15    [0]~[14]
    # 0(keep)        [15]
    # null           [16]
    # -1~-15   [17]~[31]

    blank_count=0

    for i in range(realnpy.shape[0]):
        templast = start_basic_pitch
        j = 0
        # get start pitch
        # next pitch
        # from j to length -1

        #start from first note
        while (j <= length - 1):
            if(realnpy[i][j] == 84):
                j+=1
            else:
                templast=realnpy[i][j]
                break
        j=j+1




        while (j <= length - 1):
            if (realnpy[i][j] != 84):
                cur_interval = realnpy[i][j] - templast

                if(cur_interval!=0):
                    if(is_leap_flag and (j-last_note_end<8) ): #the last one leap
                        leap_count += 1
                        if direction*cur_interval<0:
                            law_of_recovery_count+=1
                        else:
                            debug=0
                    is_leap_flag,direction=is_leap(cur_interval)


                    templast = realnpy[i][j]

                    last_note_end=j
                else:# repeat
                    last_note_end=j

            else:  # null pitch
                a=0
                # interval_array[i][j] = null_pitch

            j += 1
    return law_of_recovery_count,leap_count



def num_of_tritone(realnpy):
    start_basic_pitch = 36
    null_pitch=16

    length=realnpy.shape[1]
    interval_array=np.zeros([realnpy.shape[0],length])
    # +1~+15    [0]~[14]
    # 0(keep)        [15]
    # null           [16]
    # -1~-15   [17]~[31]
    tritone_count=0
    all_interval_count=0

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
                if(abs(cur_interval)==6):
                    tritone_count+=1
                if(cur_interval!=0):
                    all_interval_count += 0

                templast=realnpy[i][j]
            else:  #null pitch
                interval_array[i][j]=null_pitch
            j+=1
    return tritone_count,all_interval_count




def Tmain(func):

    if func=="test_law_of_recovery":
        def fun():
            file_list = []



            file="train_absolute_corpus.npy"
            gg = np.load(file)
            gg = gg[:, :]
            count1, count2 = law_of_recovery_information(gg)
            # count3=num_of_tritone()
            rate = count1 / count2
            print(gg.shape[0],str(count1) + "/" + str(count2), rate,file)

            file="obv-sf-trainabsolute.npy"
            gg=np.load(file)
            gg=gg[:,:]
            count1,count2=law_of_recovery_information(gg)
            rate=count1/count2
            print(gg.shape[0],str(count1)+"/"+str(count2),rate,file)

            file="gen_absolute_corpus.npy"
            gg = np.load("gen_absolute_corpus.npy")
            gg = gg[:, :]
            count1, count2 = law_of_recovery_information(gg)
            rate = count1 / count2
            print(gg.shape[0],str(count1) + "/" + str(count2), rate,file)
            return 0
        return fun()

Tmain("test_law_of_recovery")