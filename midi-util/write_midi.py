import numpy as np
import pretty_midi
# from p2m import  bar_length
bar_length=128
def save_singletrck_midis(bars, file_path, tempo=120.0, beat_resolution=24):
    # print(bars.shape[0], bars.shape[1], 24, bars.shape[3])
    # print(bars.shape)
    # print(bars.shape[0], bars.shape[1], 20, bars.shape[3])
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    #84-128
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
    write_piano_roll_to_midi(images_with_pause_list,program_nums=[0],is_drum=[False],filename=file_path,velocity=100,
                                         tempo=tempo, beat_resolution=beat_resolution)

# def write_piano_roll_to_midi(piano_rolls, program_nums=None, is_drum=None, filename='test.mid', velocity=100,
#                               tempo=120.0, beat_resolution=24):




def save_double_track_midis(bars, file_path, tempo=120.0, beat_resolution=24):
    # print(bars.shape[0], bars.shape[1], 24, bars.shape[3])
    # print(bars.shape)
    # print(bars.shape[0], bars.shape[1], 20, bars.shape[3])
    #tag
    # temp=bars
    # bars=np.concatenate((bars,bars),axis=3)

    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    #84-128
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
    write_piano_roll_to_midi(images_with_pause_list,program_nums=[0,0],is_drum=[False,False],filename=file_path,velocity=100,
                                         tempo=tempo, beat_resolution=beat_resolution)





def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=24):
    # Calculate time per pixel
    tpp = 60.0 / tempo / float(beat_resolution)
    # threshold = 60.0 / tempo / 4
    threshold = 0
    phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # Iterate through all possible(128) pitches

    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))
        # print('start_time:', start_time)
        # print(len(start_time))
        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))
        # print('end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('duration each note:', duration)
        # print(len(duration))

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            # print(start_time)
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                # print('i and start_time:', i, start_time[i])
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    # print(j, temp_start_time[j])
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        # print('popped start time:', temp_start_time[j])
                        t.append(j)
                        # print('popped temp_start_time:', t)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])
                # print('popped temp_start_time:', temp_start_time)

        start_time = temp_start_time
        # print('After checking, start_time:', start_time)
        # print(len(start_time))
        end_time = temp_end_time
        # print('After checking, end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('After checking, duration each note:', duration)
        # print(len(duration))

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
                # Add the note to the Instrument object
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=phrase_end_time)
                # Add the note to the Instrument object
                instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
    # print(max([i.end for i in instrument.notes]))
    # print('tpp, threshold, phrases_end_time:', tpp, threshold, phrase_end_time)


def write_piano_roll_to_midi(piano_rolls, program_nums=None, is_drum=None, filename='test.mid', velocity=100,
                              tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums have different sizes...")
        print(len(piano_rolls) ,len(program_nums))

        return False
    if not program_nums:
        program_nums = [0, 0, 0]
    if not is_drum:
        is_drum = [False, False, False]
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments

    for idx in range(len(piano_rolls)):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(program=program_nums[idx], is_drum=is_drum[idx])
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(piano_rolls[idx], instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)




def write_piano_rolls_to_midi(piano_rolls, tracks_index,program_nums=None, is_drum=None, filename='test.mid', velocity=100,
                              tempo=120.0, beat_resolution=24):
    # if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
    #     print("Error: piano_rolls and program_nums have different sizes...")
    #     return False
    # if not program_nums:
    #     program_nums = [0, 0, 0]
    # if not is_drum:
    #     is_drum = [False, False, False]

    # Create a PrettyMIDI object
    # print(tracks_index.shape)
    # print(tracks_index)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for idx in range(5):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(program=(int)(tracks_index[idx,0]), is_drum=(int)(tracks_index[idx,1]))
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(piano_rolls[idx], instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    # print(filename)
    midi.write(filename)
