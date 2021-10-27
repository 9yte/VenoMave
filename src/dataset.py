import os
import json
import math
import logging
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import recognizer.hmm as HMM
import recognizer.tools as tools
import torch
import torchaudio
from praatio import tgio
from recognizer.model import init_model


class Dataset:

    def __init__(self, data_dir, feature_parameters, seed=None, subset=None):
        
        logging.info(f"[+] Import dataset {data_dir}")
        self.feature_parameters = feature_parameters
        self.data_dir = data_dir
        self.poisons = None

        data_dir = Path(data_dir)
        # get hmm
        self.hmm = pickle.load(data_dir.parent.joinpath('hmm.h5').open('rb'))

        # get filenames 
        self.filenames = [f.stem for f in sorted(data_dir.joinpath('wavs').glob('*.wav')) ]
        
        # shuffle and subset files
        if seed is None:
            print("WARNING: dataset is not being shuffled")
        else:
            random.Random(seed + 2021).shuffle(self.filenames)

        if subset: self.filenames = self.filenames[:subset]

        self.filename_to_idx = {filename: idx for idx, filename in enumerate(self.filenames)}

        # load data
        self.wav_files = [data_dir.joinpath('wavs', f).with_suffix('.wav')
                           for f in self.filenames ]
        self.texts = [data_dir.joinpath('text', f).with_suffix('.txt').read_text()
                       for f in self.filenames ]
        self.X = [ torch.load(data_dir.joinpath('X', f).with_suffix('.pt')).cpu()
                   for f in  tqdm(self.filenames, bar_format='    X {l_bar}{bar:30}{r_bar}') ]
        self.Y = [ torch.load(data_dir.joinpath('Y', f).with_suffix('.pt')).cpu()
                   for f in  tqdm(self.filenames, bar_format='    Y {l_bar}{bar:30}{r_bar}') ]

    def get_absolute_wav_path(self, filename):
        return self.data_dir.joinpath('wavs', filename).with_suffix('.wav')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, key):
        idx = self.filenames.index(key)
        return self.wav_files[idx], self.X[idx], self.Y[idx], self.texts[idx]

    def load_wav(self, wav_file):
        return load_wav(wav_file, self.feature_parameters)

    def get_speaker_utterances(self, speaker):
        x = []
        y = []
        text = []
        files = []
        for filename in self.filenames:
            if filename.startswith(speaker):
                idx = self.filename_to_idx[filename]
                x.append(self.X[idx])
                y.append(self.Y[idx])
                text.append(self.texts[idx])
                files.append(filename)

        return files, x, y, text

    def update_y_label(self, ys, filenames):
        for y, filename in zip(ys, filenames):
            idx = self.filename_to_idx[filename]
            self.Y[idx] = y

            if self.poisons and self.poisons.is_poison(filename):
                self.poisons.update_poison_y_label(filename, y)

    def _generator(self, return_filename=False, shuffle=False, rand_obj=None, device='cuda'):
        dataset_indices = list(range(0, len(self.X)))
        
        if rand_obj:
            assert shuffle
            rand_obj.shuffle(dataset_indices)
        elif shuffle:
            random.shuffle(dataset_indices)
        
        for dataset_index in dataset_indices:
            x, y, text, filename = self.X[dataset_index], self.Y[dataset_index], self.texts[dataset_index], self.filenames[dataset_index]
            # overwrite with current value if poisoned
            if self.poisons and self.poisons.is_poison(filename):
                x = self.poisons.get_poison(filename)
            
            x = x.to(device)
            y = y.to(device)
            
            if return_filename:
                yield x, y, text, filename
            else:
                yield x, y, text

    def generator(self, batch_size=1, return_filename=False, return_x_length=False, shuffle=False, rand_obj=None, device='cuda'):
        no_of_batches = math.ceil(len(self.filenames) / batch_size)
        samples = self._generator(return_filename=return_filename, shuffle=shuffle, rand_obj=rand_obj, device=device)
        for batch_idx in range(no_of_batches):
            if batch_idx == no_of_batches - 1 and len(self.filenames) % batch_size != 0:
                batch = [list(a) for a in zip(*[next(samples) for _ in range(len(self.filenames) % batch_size)])]
            else:
                batch = [list(a) for a in zip(*[next(samples) for _ in range(batch_size)])]

            X = batch[0]
            Y = batch[1]
            TEXT = batch[2]

            if batch_size == 1:
                yield X[0], Y[0].long(), TEXT[0], len(Y[0])
            else:
                x_batch_true_length = [x.shape[1] for x in X]
                max_x_l = max(x_batch_true_length)
                x_batch = [F.pad(x, pad=(0, max_x_l - x.shape[1], 0, 0)) for x in X]
                x_batch = torch.cat(x_batch)

                y_batch_true_length = [len(y) for y in Y]
                max_y_l = max(y_batch_true_length)

                # We pad y_batch with labels of SILENCE, which should be [1,0,0,0,...]
                silence = [1] + [0] * (Y[0].shape[1] - 1)
                silence = torch.tensor(silence).reshape(1, -1).to(Y[0].device)

                for index, y in enumerate(Y):
                    z = silence.repeat((max_y_l - len(y), 1))
                    Y[index] = torch.cat((y, z))
                y_batch = torch.stack(Y)

                if return_filename:
                    if return_x_length:
                        yield x_batch, y_batch, TEXT, y_batch_true_length, x_batch_true_length, batch[3]
                    else:
                        yield x_batch, y_batch, TEXT, y_batch_true_length, batch[3]
                else:
                    if return_x_length:
                        yield x_batch, y_batch, TEXT, y_batch_true_length, x_batch_true_length
                    else:
                        yield x_batch, y_batch, TEXT, y_batch_true_length

    def word_states_mapping(self):
        """
        Creates a mapping, which helps to determine states of each word!
        :param hmm:
        :return:
        """
        assert self.hmm.mode == 'word'

        word_to_states = {}
        state_to_word = {}
        start_idx = 0
        for word, size in zip(self.hmm.words['name'], self.hmm.words['size']):
            word_to_states[word] = list(range(start_idx, start_idx + size))
            for idx in range(start_idx, start_idx + size):
                state_to_word[idx] = word
            start_idx += size
        return word_to_states, state_to_word

    def phoneme_states_freq(self):
        """
        This module collect statistics about each state of the phoneme in the dataset!
        How frequent each phoneme or word/digit occurs in the dataset.
        We use this later on during selecting adv. states and accordingly poisons!
        :return:
        """
        data_dir = self.data_dir
        phoneme_freq_path = "{}/phoneme_freq.json".format(data_dir)
        if os.path.exists(phoneme_freq_path):
            with open(phoneme_freq_path) as f:
                freqs = json.load(f)

                states_ratio = freqs['states_ratio']
                states_ratio = {int(k): v for k, v in states_ratio.items()}

                states_freq = freqs["states_freq"]
                states_freq = {int(k): v for k, v in states_freq.items()}

        else:
            word_to_states, state_to_word = self.word_states_mapping()

            words_freq = {}
            states_freq = {}
            for word, states in word_to_states.items():
                # word_digit = tools.WORD_TO_DIGIT[word]
                words_freq[word] = 0
                for state in states:
                    states_freq[state] = 0

            for y, text in zip(self.Y, self.texts):
                for word in text.split(" "):
                    word = word.lower()
                    # word_digit = tools.WORD_TO_DIGIT[word]
                    words_freq[word] += 1
                y = torch.argmax(y, dim=1)
                for state in y:
                    states_freq[state.item()] += 1

            states_ratio = {state: (1.0 * state_freq) / words_freq[state_to_word[state]]
                            for state, state_freq in states_freq.items() if state != 0}

            with open(phoneme_freq_path, 'w') as f:
                json.dump({'states_ratio': states_ratio, "states_freq": states_freq, "words_freq": words_freq}, f)

        return states_ratio, states_freq


def load_wav(wav_file, feature_parameters):
    x, _ = torchaudio.load(wav_file)
    # round to the next `full` frame
    num_frames = np.floor(x.shape[1]/feature_parameters['hop_size_samples'])
    return x[:,:int(num_frames * feature_parameters['hop_size_samples'])].cuda()


def preprocess_dataset(task, model_type, data_dir, feature_parameters, speakers_list, speakers_split_identifier,
                       only_plain=False, device='cpu'):

    def load_raw_data_dir(dataset_dir, speakers_list=None, device='cpu'):
        dataset_dir = dataset_dir.resolve()  # To resolve symlinks!
        # find raw data
        wav_files = [f for f in sorted(dataset_dir.joinpath('wav').resolve().glob('*.wav'))]
        praat_files = [f for f in sorted(dataset_dir.joinpath('TextGrid').resolve().glob('*.TextGrid'))]
        lab_files = [f for f in sorted(dataset_dir.joinpath('lab').resolve().glob('*.lab'))]

        # load raw data
        X = []
        Y = []
        texts = []
        wav_files_selected = []
        for wav_file, praat_file, lab_file in tqdm(zip(wav_files, praat_files, lab_files), 
                                                   total=len(wav_files), bar_format='    load raw     {l_bar}{bar:30}{r_bar}'):
            # sanity check
            assert wav_file.stem == praat_file.stem == lab_file.stem
            if speakers_list and '-'.join(wav_file.stem.split("-")[-3:-1]) not in speakers_list:
                continue
            wav_files_selected.append(wav_file)
            ## load x
            x, _ = torchaudio.load(wav_file)
            # round to the next `full` frame
            num_frames = np.floor(x.shape[1]/hop_size_samples)
            x = x[:,:int(num_frames * hop_size_samples)].to(device)
            X.append(x)
            ## load y
            # optional: convert praats into jsons
            # dataset_dir.joinpath('align').mkdir(parents=True, exist_ok=True)
            # tg = tgio.openTextgrid(praat_file)
            # align_dict = tools.textgrid_to_dict(tg)
            # json_file = Path(str(praat_file).replace('TextGrid', 'align')).with_suffix('.json')
            # json_file.write_text(json.dumps(align_dict, indent=4))
            # y = tools.json_file_to_target(json_file, sampling_rate, window_size_samples, hop_size_samples, hmm)
            y = tools.praat_file_to_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm)		
            y = torch.from_numpy(y).to(device)
            Y.append(y)
            ## load text
            text = lab_file.read_text().strip()
            texts.append(text)
        return wav_files_selected, X, Y, texts

    """
    Creates two datasets:
        - plain is simply a pre-processed version of TIDIGITS 
        - aligned replaces the targets Y with more precise targets (obtained via viterbi training)

    """

    if True:
        # check if data dir exist
        raw_data_dir = Path(data_dir).joinpath('raw')
        assert raw_data_dir.is_dir()

        # data config
        sampling_rate = feature_parameters['sampling_rate']
        window_size_samples = tools.next_pow2_samples(feature_parameters['window_size'], sampling_rate)
        hop_size_samples = tools.sec_to_samples(feature_parameters['hop_size'], sampling_rate)

        # check if dataset is already pre-processed
        if speakers_split_identifier in ['speakers-all', 'speakers-none']:
            plain_out_dir = Path(data_dir).joinpath(model_type, 'plain')
            aligend_out_dir = Path(data_dir).joinpath(model_type, 'aligned')
        else:
            plain_out_dir = Path(data_dir).joinpath(f'{model_type}-{speakers_split_identifier}', 'plain')
            aligend_out_dir = Path(data_dir).joinpath(f'{model_type}-{speakers_split_identifier}', 'aligned')

        if plain_out_dir.joinpath('hmm.h5').is_file() and aligend_out_dir.joinpath('hmm.h5').is_file():
            logging.info(f"[+] Dataset already pre-processed")
            return
        shutil.rmtree(plain_out_dir,ignore_errors=True); plain_out_dir.mkdir(parents=True)
        shutil.rmtree(aligend_out_dir,ignore_errors=True); aligend_out_dir.mkdir(parents=True)

        # Step 1: plain data
        # -> wavs are split into individual frames (the Xs)
        # -> each frame is mapped to the corresponding target state
        #    of the hmm (the Ys)
        #
        # As these targets are always depend on a particular hmm,
        # we save the hmm alongside with the data
        hmm = HMM.HMM(task, 'word')
        pickle.dump(hmm, plain_out_dir.joinpath('hmm.h5').open('wb'))

        # pre-proccess plain data
        dataset_names = [ d.name for d in Path(raw_data_dir).glob('*') if d.is_dir() ]
        for dataset_name in dataset_names:
            logging.info(f"[+] Pre-process {dataset_name}")
            if 'train' in dataset_name.lower():
                if speakers_list is not None:
                    speakers_list = set(speakers_list)
                wav_files, X, Y, texts = load_raw_data_dir(raw_data_dir.joinpath(dataset_name), speakers_list)
            elif 'test' == dataset_name.lower():
                wav_files, X, Y, texts = load_raw_data_dir(raw_data_dir.joinpath(dataset_name))
            else:
                pass
            ## dump plain
            X_out_dir = plain_out_dir.joinpath(dataset_name, 'X'); X_out_dir.mkdir(parents=True)
            Y_out_dir = plain_out_dir.joinpath(dataset_name, 'Y'); Y_out_dir.mkdir(parents=True)
            text_out_dir = plain_out_dir.joinpath(dataset_name, 'text'); text_out_dir.mkdir(parents=True)
            wav_out_dir = plain_out_dir.joinpath(dataset_name, 'wavs'); wav_out_dir.mkdir(parents=True)
            for wav_file, x, y, text in tqdm(zip(wav_files, X, Y, texts), 
                                             total=len(wav_files), bar_format='    dump plain  {l_bar}{bar:30}{r_bar}'):
                filename = wav_file.stem
                torch.save(y, Y_out_dir.joinpath(filename).with_suffix('.pt'))
                torch.save(x, X_out_dir.joinpath(filename).with_suffix('.pt'))
                text_out_dir.joinpath(filename).with_suffix('.txt').write_text(text)
                shutil.copyfile(wav_file, wav_out_dir.joinpath(filename).with_suffix('.wav'))

        if only_plain:
            return
        else:
            # Step 2: align data
            # -> for the plain data we only used relatively vague alignements between
            #    input frame and target
            # -> to improve this we create a second dataset that uses a hmm
            #    that is trained with viterbi to obtain more precise alignments

            # first we need to train the hmm with viterbi training
            dataset = Dataset(plain_out_dir.joinpath('TRAIN'), feature_parameters, seed=8734)
            model = init_model(model_type, feature_parameters, hmm)
            model.train_model(dataset, epochs=15, batch_size=32)
            model.train_model(dataset, epochs=1, batch_size=128, viterbi_training=True)
            pickle.dump(hmm, aligend_out_dir.joinpath('hmm.h5').open('wb'))
            model.hmm.A = hmm.modifyTransitions(model.hmm.A_count)
            # model.train_model(dataset, epochs=2, batch_size=128, viterbi_training=True)
            # again, save hmm alongside the data
            pickle.dump(hmm, aligend_out_dir.joinpath('hmm.h5').open('wb'))

    else:
        # In case, we have the aligned hmm saved, but aligned states are not saved yet!!!
        raw_data_dir = Path(data_dir).joinpath('raw')
        assert raw_data_dir.is_dir()
        plain_out_dir = Path(data_dir).joinpath(model_type, 'plain')
        aligend_out_dir = Path(data_dir).joinpath(model_type, 'aligned')

        dataset = Dataset(plain_out_dir.joinpath('TRAIN'), feature_parameters, seed=8734)
        hmm = HMM.HMM(task, 'word')
        model = init_model(model_type, feature_parameters, hmm)
        model.train_model(dataset, epochs=15, batch_size=32)
        hmm = pickle.load(aligend_out_dir.joinpath('hmm.h5').open('rb'))
        model.hmm = hmm

    # pre-proccess aligned data
    dataset_names = [ d.name for d in Path(raw_data_dir).glob('*') if d.is_dir() ]
    for dataset_name in dataset_names:
        logging.info(f"[+] Pre-process {dataset_name}")
        # wav_files, X, Y, texts = load_raw_data_dir(raw_data_dir.joinpath(dataset_name), device=device)
        dst_path = plain_out_dir.joinpath(dataset_name)
        dataset = Dataset(dst_path, feature_parameters, seed=8735)
        ## dump plain
        X_out_dir = aligend_out_dir.joinpath(dataset_name, 'X'); X_out_dir.mkdir(parents=True, exist_ok=True)
        Y_out_dir = aligend_out_dir.joinpath(dataset_name, 'Y'); Y_out_dir.mkdir(parents=True, exist_ok=True)
        text_out_dir = aligend_out_dir.joinpath(dataset_name, 'text'); text_out_dir.mkdir(parents=True, exist_ok=True)
        wav_out_dir = aligend_out_dir.joinpath(dataset_name, 'wavs'); wav_out_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(dataset.wav_files), bar_format='    dump aligned {l_bar}{bar:30}{r_bar}') as pbar:
            for X_batch, Y_batch, texts_batch, y_true_length, x_true_length, filenames in dataset.generator(return_filename=True, batch_size=128, return_x_length=True):
                for filename in filenames:
                    if not X_out_dir.joinpath(filename).with_suffix('.pt').exists():
                        # means this batch has at least one file, which is not yet aligned!
                        # let's process the whole batch, and then break!
                        
                        posteriors = model.features_to_posteriors(X_batch)
                        Y_batch = hmm.viterbi_train(posteriors, y_true_length, Y_batch, texts_batch, n_jobs=64)

                        for filename, x, y, y_length, x_length, text in zip(filenames, X_batch, Y_batch, y_true_length, x_true_length, texts_batch):
                            torch.save(y.clone()[:y_length], Y_out_dir.joinpath(filename).with_suffix('.pt'))
                            torch.save(x.clone()[:x_length].unsqueeze(dim=0), X_out_dir.joinpath(filename).with_suffix('.pt'))
                            text_out_dir.joinpath(filename).with_suffix('.txt').write_text(text)
                            shutil.copyfile(dst_path.joinpath('wavs', filename).with_suffix('.wav'), wav_out_dir.joinpath(filename).with_suffix('.wav'))

                        break
                pbar.update(len(filenames))
