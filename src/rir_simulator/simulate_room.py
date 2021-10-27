import roomsimove_single
import soundfile as sf
import olafilt
import numpy as np



rt60 = 0.4 # reveberation time in seconds
room_dim = [4.2, 3.4, 5.2] # room dimension in meters
mic_pos1 = [2, 2, 2]; mic_pos2 = [2, 2, 2] # microphone position in meters
source_pos = [1, 1, 1] # speakers position in  meters
sampling_rate = 16000

mic_positions = [mic_pos1, mic_pos2]
rir = roomsimove_single.do_everything(room_dim, mic_positions, source_pos, rt60) # get room impulse response


[data, fs] = sf.read('TRAIN-MAN-AE-2B.poison.wav') # ,always_2d=True)
data_rev = olafilt.olafilt(rir[:,0], data)

sf.write('TRAIN-MAN-AE-2B.poison.reverb.wav', data_rev.T, fs)