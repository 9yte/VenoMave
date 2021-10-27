import numpy as np
from praatio import tgio
import random
from pathlib import Path
import torch
import torchaudio
import shutil
import os
import logging
import json

from collections import defaultdict


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
WORD_TO_DIGIT = {w: i for i, w in enumerate(WORDS)}
DIGIT_TO_WORD = {str(i): w for i, w in enumerate(WORDS) if i > 0}
WORDS.append("oh")
WORD_TO_DIGIT["oh"] = 'O'
DIGIT_TO_WORD['O'] = 'oh'
DIGIT_TO_WORD['Z'] = 'zero'
WORD_TO_DIGIT['zero'] = 'Z'
WORD_TO_DIGIT['sil'] = 'S'
DIGIT_TO_WORD['S'] = 'sil'


def digits_to_str(digits):
    return [DIGIT_TO_WORD[digit] for digit in digits]


def str_to_digits(strs):
    return [WORD_TO_DIGIT[s.lower()] for s in strs]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sec_to_samples(x, sampling_rate):
    """
    Converts continuous time to sample index.

    :param x: scalar value representing a point in time in seconds.
    :param samskypling_rate: sampling rate in Hz.
    :return: sample index.
    """
    return int(x * sampling_rate)


def next_pow2(x):
    """
    Returns the next power of two for any given positive number.

    :param x: scalar input number.
    :return: next power of two larger than input number.
    """
    return int(np.ceil(np.log2(np.abs(x))))


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """
    Returns the total number of frames for a given signal length with corresponding window and hop sizes.

    :param signal_length_samples: total number of samples.
    :param window_size_samples: window size in samples.
    :param hop_size_samples: hop size (frame shift) in samples.
    :return: total number of frames.
    """
    overlap_samples = window_size_samples - hop_size_samples

    return int(np.ceil(float(signal_length_samples + 1 - overlap_samples) / hop_size_samples))


def hz_to_mel(x):
    """
    Converts a frequency given in Hz into the corresponding Mel frequency.

    :param x: input frequency in Hz.
    :return: frequency in mel-scale.
    """
    return 2595 * np.log10(1 + float(x) / 700)


def mel_to_hz(x):
    """
    Converts a frequency given in Mel back into the linear frequency domain in Hz.

    :param x: input frequency in mel.
    :return: frequency in Hz.
    """
    return 700 * (10 ** (x / 2595) - 1)


def next_pow2_samples(x, sampling_rate):
    """
    Returns the next power of two in number of samples for a given length in seconds and sampling rate

    :param x: length in seconds.
    :sampling_rate: sampling rate in Hz.
    :return: next larger power of two in number of samples
    """
    # return 2**next_pow2(sec_to_samples(x, sampling_rate))
    return sec_to_samples(x, sampling_rate)


def sample_to_frame(x, window_size_samples, hop_size_samples):
    """
    converts sample index to frame index.

    :param x: sample index.
    :param window_size_samples: window length in samples.
    :param hop_size_samples:    hop length in samples
    :return: frame index.
    """
    return int(np.floor(x / hop_size_samples))


def sec_to_frame(x, sampling_rate, window_size_samples, hop_size_samples):
    """
    Converts time in seconds to frame index.

    :param x:  time in seconds
    :param sampling_rate:  sampling frequency in hz
    :param window_size_samples: window length in samples.
    :param hop_size_samples:    hop length in samples
    :return: frame index
    """
    return sample_to_frame(sec_to_samples(x, sampling_rate), window_size_samples, hop_size_samples)


def divide_interval(num, start, end):
    """
    Divides the number of states equally to the number of frames in the interval.

    :param num:  number of states.
    :param start: start frame index
    :param end: end frame index
    :return starts: start indexes
    :return end: end indexes
    """
    interval_size = end - start
    # gets remainder 
    remainder = interval_size % num
    # init sate count per state with min value
    count = [int((interval_size - remainder) / num)] * num
    # the remainder is assigned to the first n states
    count[:remainder] = [x + 1 for x in count[:remainder]]
    # init starts with first start value
    starts = [start]
    ends = []
    # iterate over the states and sets start and end values
    for c in count[:-1]:
        ends.append(starts[-1] + c)
        starts.append(ends[-1])

    # set last end value
    ends.append(starts[-1] + count[-1])

    return starts, ends


def praat_file_to_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm, word=True):
    """
    Reads in praat file and calculates the phone-based target matrix.

    :param praat_file: *.TextGrid file.
    :param sampling_rate: sampling frequency in hz
    :param window_size_samples: window length in samples
    :param hop_size_samples: hop length in samples
    :return: target matrix for DNN training
    """
    # gets list of intervals, start, end, and word/phone
    if hmm.mode == 'word':
        intervals, min_time, max_time = praat_to_word_interval(praat_file)
    elif hmm.mode == 'phoneme':
        # TODO need to fix this if we want to use json
        intervals, min_time, max_time = praat_to_phone_interval(praat_file)

    # we assume min_time always to be 0, if not, we have to take care of this
    if not min_time == 0:
        raise Exception("Houston we have a problem: start value of audio file is not 0 for file: {}".format(praat_file))

    # gets dimensions of target
    max_sample = sec_to_samples(max_time, sampling_rate)
    num_frames = get_num_frames(max_sample, window_size_samples, hop_size_samples)
    num_states = hmm.get_num_states()

    # init target with zeros
    target = np.zeros((num_frames, num_states))

    # parse intervals
    for interval in intervals:
        # get state index, start and end frame
        states = hmm.input_to_state(interval.label)
        start_frame = sec_to_frame(interval.start, sampling_rate, window_size_samples, hop_size_samples)
        end_frame = sec_to_frame(interval.end, sampling_rate, window_size_samples, hop_size_samples)

        # divide the interval equally to all states
        starts, ends = divide_interval(len(states), start_frame, end_frame)

        # assign one-hot-encoding to all segments of the interval
        for state, start, end in zip(states, starts, ends):
            # set state from start to end to 1
            target[start:end, state] = 1

    # find all columns with only zeros...
    zero_column_idxs = np.argwhere(np.amax(target, axis=1) == 0)
    # ...and set all as silent state
    target[zero_column_idxs, hmm.input_to_state('sil')] = 1

    # if "target" in Path(praat_file).as_posix():
    #     target_labels = np.argmax(target, 1)

    #     idx = (target_labels == 33).nonzero()
    #     target_labels[idx] = 51

    #     target_new = np.zeros(target.shape)
    #     target_new[np.arange(target.shape[0]), target_labels] = 1

    #     target = target_new

    return target


def praat_file_to_phone_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm):
    """
    Reads in praat file and calculates the phone-based target matrix.

    :param praat_file: *.TextGrid file.
    :param sampling_rate: sampling frequency in hz
    :param window_size_samples: window length in samples
    :param hop_size_samples: hop length in samples
    :return: target matrix for DNN training
    """
    # gets list of intervals, start, end, and word/phone
    intervals, min_time, max_time = praat_to_phone_Interval(praat_file)

    # we assume min_time always to be 0, if not, we have to take care of this
    if not min_time == 0:
        raise Exception("Houston we have a problem: start value of audio file is not 0 for file: {}".format(praat_file))

    # gets dimensions of target
    max_sample = sec_to_samples(max_time, sampling_rate)
    num_frames = get_num_frames(max_sample, window_size_samples, hop_size_samples)
    num_states = hmm.get_num_states()
    # init target with zeros
    target = np.zeros((num_frames, num_states))

    # parse intervals
    for interval in intervals:
        # get state index, start and end frame
        states = hmm.input_to_state(interval.label)
        start_frame = sec_to_frame(interval.start, sampling_rate, window_size_samples, hop_size_samples)
        end_frame = sec_to_frame(interval.end, sampling_rate, window_size_samples, hop_size_samples)

        # divide the interval equally to all states
        starts, ends = divide_interval(len(states), start_frame, end_frame)

        # assign one-hot-encoding to all segments of the interval
        for state, start, end in zip(states, starts, ends):
            # set state from start to end to 1
            target[start:end, state] = 1

    # find all columns with only zeros...
    zero_column_idxs = np.argwhere(np.amax(target, axis=1) == 0)
    # ...and set all as silent state
    target[zero_column_idxs, hmm.input_to_state('sil')] = 1

    return target


def praat_to_word_interval(praat_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.TextGrid file path

    :return itervals: returns list of intervals, 
                        containing start and end time and the corresponding word/phobe.
    :return min_time:    min timestamp of audio (should be 0)
    :return max_time:    min timestamp of audio (should be audio length)
    """
    # read in praat file (expects one *.TextGrid file path)
    tg = tgio.openTextgrid(praat_file)

    # read return values
    itervals = tg.tierDict['words'].entryList
    min_time = tg.minTimestamp
    max_time = tg.maxTimestamp

    # we will read in word-based
    return itervals, min_time, max_time


class Interval:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label


def json_to_word_interval(json_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.json file path

    :return itervals: returns list of intervals, 
                        containing start and end time and the corresponding word/phobe.
    :return min_time:    min timestamp of audio (should be 0)
    :return max_time:    min timestamp of audio (should be audio length)
    """
    # read in praat file (expects one *.json file path)
    align_dict = json.loads(json_file.read_text())

    # read return values
    itervals = []
    for _, entry_dict in align_dict['words'].items():
        interval = Interval(entry_dict['start'], entry_dict['end'], entry_dict['label'])
        itervals.append(interval)

    min_time = align_dict['min_time']
    max_time = align_dict['max_time']

    # we will read in word-based
    return itervals, min_time, max_time


def praat_to_phone_Interval(praat_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.TextGrid file path

    :return itervals: returns list of intervals, 
                        containing start and end time and the corresponding word/phone.
    :return min_time: min timestamp of audio (should be 0)
    :return max_time: min timestamp of audio (should be audio length)
    """
    # read in praat file (expects one *.TextGrid file path)
    tg = tgio.openTextgrid(praat_file)

    # read return values
    itervals = tg.tierDict['phones'].entryList
    min_time = tg.minTimestamp
    max_time = tg.maxTimestamp

    # we will read in word-based
    return itervals, min_time, max_time


def shuffle_list(*ls):
    """
    Shuffles all list in ls with same permutation

    :param ls: list of list to shuffle.
    :return: shuffled lists.
    """
    l = list(zip(*ls))

    random.shuffle(l)
    return zip(*l)


def load_label(y_praat, hmm, sampling_rate, parameters, device='cpu', word=True):
    # same values for all utterances
    window_size_samples = next_pow2_samples(parameters['window_size'], sampling_rate)
    hop_size_samples = sec_to_samples(parameters['hop_size'], sampling_rate)

    y = json_file_to_target(y_praat, sampling_rate, window_size_samples, hop_size_samples, hmm)

    return torch.from_numpy(np.argmax(y, 1)).to(device)


# def load_targetaudio(wav_path, parameters, sampling_rate, hmm, normalize_x=True, device='cpu', word=True):

#     hop_size_samples = tools.sec_to_samples(parameters['hop_size'], sampling_rate)
#     window_size_samples = tools.next_pow2_samples(parameters['window_size'], sampling_rate)
#    # tar_label = tools.praat_file_to_target(target_dict['original_tgrid_path'], sampling_rate, window_size_samples, hop_size_samples, hmm, word=word)

#     x, _ = torchaudio.load(wav_path) 

#     # num_frames = np.floor(x.shape[1] / hop_size_samples)
#     # x = x[:, :int(num_frames * hop_size_samples) - 1]

#     num_frames = tools.get_num_frames(x.shape[1], window_size_samples, hop_size_samples) + 1

#     # if target length does not fit signal length
#     # signal_length = x.shape[1]
#     # if tar_label.shape[0] != num_frames:
#     #     x = x[:, :-hop_size_samples]
#     #     signal_length = x.shape[1]
#     #     num_frames = tools.get_num_frames(signal_length, window_size_samples, hop_size_samples) + 1

#     if normalize_x:
#         x = x / torch.max(torch.abs(x))

#     return x.to(device)


def get_sampling_rate(data_dir):
    train_dir = Path(data_dir, 'raw', 'TRAIN')
    x = list(Path(train_dir, 'wav').glob('*.wav'))[0]
    _, sampling_rate = torchaudio.load(x)
    return sampling_rate


def get_seed(model_idx, crafting_step, init_seed):
    # We assume the model_idx and crafting_step are smaller than 500!
    assert model_idx < 500 and crafting_step < 500
    return init_seed + 500 * model_idx + crafting_step


# def increment_one(num_str):
#     if num_str == "OH":
#         return "ONE"
#     else:
#         i = WORD_DIGIT[num_str]
#         i = (i + 1) % len(WORDS)
#         return WORDS[i]


def textgrid_to_dict(textgrid):
    align_dict = defaultdict(lambda: defaultdict(dict))

    for key, value in textgrid.tierDict.items():
        for idx, interval in enumerate(value.entryList):
            align_dict[key][idx]['start'] = interval.start
            align_dict[key][idx]['end'] = interval.end
            align_dict[key][idx]['label'] = interval.label

    align_dict['min_time'] = textgrid.minTimestamp
    align_dict['max_time'] = textgrid.maxTimestamp

    return dict(align_dict)


def numel(array):
    s = array.shape
    n = 1
    for i in range(len(s)):
        n *= s[i]

    return n


def snrseg(noisy, clean, fs, tf=0.05):
    '''
    Segmental SNR computation. Does NOT support VAD or Interpolation (at the moment). Corresponds to the mode 'wz' in
    the original Matlab implementation.

    SEG = mean(10*log10(sum(Ri^2)/sum((Si-Ri)^2))

    '''

    snmax = 100
    noisy = noisy.squeeze()
    clean = clean.squeeze()

    nr = min(clean.shape[0], noisy.shape[0])
    kf = round(tf * fs)
    ifr = np.arange(kf, nr, kf)
    ifl = int(ifr[len(ifr) - 1])
    nf = numel(ifr)

    ef = np.sum(np.reshape(np.square((noisy[:ifl] - clean[:ifl])), (kf, nf), order='F'), 0)
    rf = np.sum(np.reshape(np.square(clean[:ifl]), (kf, nf), order='F'), 0)

    em = ef == 0
    rm = rf == 0

    snf = 10 * np.log10((rf + rm) / (ef + em))
    snf[rm] = -snmax
    snf[em] = snmax

    # Equivalent for matlab true(1,nf):
    temp = np.ones(nf)
    vf = temp == 1

    seg = np.mean(snf[vf])
    # glo = 10 * np.log10(sum(rf[vf]) / sum(ef[vf]))

    return seg