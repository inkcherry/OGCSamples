from __future__ import print_function
import os
import json
import errno
from pypianoroll import Multitrack, Track
import pretty_midi
import shutil
import numpy as np
import write_midi
import  tensorflow as tf
#config
ROOT_PATH = '/home/liumingzhi/inkcpro/midilib/lib'
converter_path = os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/converter')
cleaner_path = os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner')
bar_length=32


#

def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not
    exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_midi_path(root):
    """Return a list of paths to MIDI files in `root` (recursively)"""
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mid'):
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths


def get_midi_info(pm):
    """Return useful information from a pretty_midi.PrettyMIDI instance"""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info


def midi_filter(midi_info):
    """Return True for qualified midi files and False for unwanted ones"""
    if midi_info['first_beat_time'] > 0.0:
        return False
    elif midi_info['num_time_signature_change'] > 1:
        return False
    elif midi_info['time_signature'] not in ['4/4']:
        return False
    return True


def get_merged(multitrack):
    """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
    five tracks (Bass, Drums, Guitar, Piano and Strings)"""
    category_list = {'Piano': []}
    program_dict = {'Piano': 0}

    for idx, track in enumerate(multitrack.tracks):

        if track.program//8 == 0:
            category_list['Piano'].append(idx)


    tracks = []

    for key in category_list:
        if category_list[key]:
            merged = multitrack[category_list[key]].get_merged_pianoroll('max')

            tracks.append(Track(merged, program_dict[key], key == 'Drums', key))
        else:
            tracks.append(Track(None, program_dict[key], key == 'Drums', key))
    return tracks,Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat, multitrack.beat_resolution, multitrack.name)


def converter(filepath):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`"""
    try:
        midi_name = os.path.splitext(os.path.basename(filepath))[0]
        multitrack = Multitrack(beat_resolution=24, name=midi_name)

        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = get_midi_info(pm)
        multitrack.parse_pretty_midi(pm)
        merged = get_merged(multitrack)

        make_sure_path_exists(converter_path)
        merged.save(os.path.join(converter_path, midi_name + '.npz'))

        return [midi_name, midi_info]

    except:
        return None

def get_bar_piano_roll(piano_roll,last_bar_mode='remove'):
    if int(piano_roll.shape[0] % bar_length) is not 0:
        if last_bar_mode == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((bar_length - piano_roll.shape[0] % bar_length, 128))), axis=0)
        elif last_bar_mode == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % bar_length):], axis=0)
    piano_roll = piano_roll.reshape(-1, bar_length, 128)
    return piano_roll


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], bar_length, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, bar_length, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                 images_with_pause.shape[1],
                                                                                 images_with_pause.shape[2]))
    # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)

    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,
                                         tempo=tempo, beat_resolution=4)

def save_multitrack_midis(multitrack_with_bars,tracks_index,file_path,tempo=80.0):
    multi_images_with_pause=np.zeros((5, 32, bar_length, 128, 1))
    for i in range (5):
        bars=multitrack_with_bars[i]
        # print("bars shape",bars.shape)
        padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                      np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
        pause = np.zeros((bars.shape[0], bar_length, 128, bars.shape[3]))
        images_with_pause = padded_bars
        images_with_pause = images_with_pause.reshape(-1, bar_length, padded_bars.shape[2], padded_bars.shape[3])
        images_with_pause_list = []
        for ch_idx in range(padded_bars.shape[3]):
            images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                     images_with_pause.shape[1],
                                                                                     images_with_pause.shape[2]))
        # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],
        #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
        multi_images_with_pause[i]=images_with_pause
    print("multi_images_shape",multi_images_with_pause.shape)

    write_midi.write_piano_rolls_to_midi(multi_images_with_pause, tracks_index,program_nums=5, is_drum=[False],
                                             filename=file_path,
                                             tempo=tempo, beat_resolution=4)







# (piano_roll, filename, 0, is_drum=False, velocity=100,
#                              tempo=120.0, beat_resolution=16):





multitrack = Multitrack(beat_resolution=4, name='exghample')
x = pretty_midi.PrettyMIDI('monophonicdata/monomelody1.mid')
multitrack.parse_pretty_midi(x)
tracks,_=get_merged(multitrack)





# tracks_index=np.zeros([5,2])  #program  is_drum
# index = 0
# for key in tracks:
#     tracks_index[index,0]=key.program
#     tracks_index[index,1]=key.is_drum
#     index+=1



# print (tracks_index)


def get_initial():
    for key in tracks:
        pr=get_bar_piano_roll(key.pianoroll)
        pr_clip = pr[:, :, 24:108]
        if int(pr_clip.shape[0] % 4) != 0:
            pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
        pr_re = pr_clip.reshape(-1, bar_length, 84, 1)
        print(pr_re.shape)
        dims=pr_re.shape
        merges=pr_re
        return dims,merges

dims,merges=get_initial()

print(dims)
idx=0
for key in tracks:
    if idx==0:
        idx+=1
    else:
        pr=get_bar_piano_roll(key.pianoroll)
        pr_clip = pr[:, :, 24:108]
        if int(pr_clip.shape[0] % 4) != 0:
            pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
        pr_re = pr_clip.reshape(-1, bar_length, 84, 1)
        merges=np.append(merges,pr_re)


merges=merges.reshape(dims[0],dims[1],dims[2],dims[3])
print(merges.shape)

write_midi.save_singletrck_midis(merges,'qws.mid')


print(merges.shape)



