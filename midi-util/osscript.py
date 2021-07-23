import os
from shutil import copyfile
import  re
ROOT_PATH="/home/liumingzhi/inkcpro/lead-sheet-dataset-master/datasets/pianoroll"
SAVE_PATH="/home/liumingzhi/inkcpro/midilib/util/leadersheetdataset/allmididataset"
NO_KEY_PATH="/home/liumingzhi/inkcpro/midilib/util/leadersheetdataset/nokeymidi"
KEY_PATH="/home/liumingzhi/inkcpro/midilib/util/leadersheetdataset/keymidi"
# ROOT_PATH="F:/inkcgit/2020.5/lead-sheet-dataset-master/datasets/pianoroll"
# SAVE_PATH="F:/inkcgit/2020.5/lead-sheet-dataset-master/mididatasets"

def get_midi():
    count=0
    first_names=[f for f in os.listdir(ROOT_PATH)]
    for first_name in first_names:
        first_name_path=os.path.join(ROOT_PATH,first_name)
        singers=[f for f in os.listdir(first_name_path)]
        for singer in singers:
            songs_path=os.path.join(first_name_path,singer)
            songs = [f for f in os.listdir(songs_path)]
            for song in songs:
                song_path=os.path.join(songs_path,song)
                files=[f for f in os.listdir(song_path)]
                for filename in files:
                    name,suffix = os.path.splitext(filename)
                    if(suffix==".mid"):
                        count+=1
                        file_path=os.path.join(song_path,filename)
                        save_name=str(count)+filename
                        save_path=os.path.join(SAVE_PATH,save_name)
                        print(count,file_path,save_path)
                        copyfile(file_path,save_path)

def get_no_key_midi():
    count_no_key=0
    count_key=0
    file_names=[f for f in os.listdir(SAVE_PATH)]
    if not os.path.exists(NO_KEY_PATH):
        os.mkdir(NO_KEY_PATH)
    if not os.path.exists(KEY_PATH):
        os.mkdir(KEY_PATH)



    for filename in file_names:
        name, suffix = os.path.splitext(filename)
        pattern=r"nokey"
        dd=re.search(pattern,name)
        if( (re.search(pattern,name))!=None ):   #get Ckey midi
            count_no_key += 1
            save_name=str(count_no_key)+"--"+filename
            save_path=os.path.join(NO_KEY_PATH,save_name)
            file_path=os.path.join(SAVE_PATH,filename)
            print(count_no_key, file_path, save_path)
            copyfile(file_path, save_path)
        else:
            count_key += 1
            save_name = str(count_key) + "--" + filename
            save_path = os.path.join(KEY_PATH, save_name)
            file_path = os.path.join(SAVE_PATH, filename)
            print(count_key, file_path, save_path)
            copyfile(file_path, save_path)

get_no_key_midi()

