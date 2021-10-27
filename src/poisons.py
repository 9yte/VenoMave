import random
import logging
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import recognizer.tools as tools
from psycho import Psycho
import torch
import torchaudio
from itertools import chain
import torch.nn.functional as F

from pathlib import Path

import json


def compute_scale_grads(poison_filename, orig_signal, modified_signal, thresh_file, psycho, device='cuda'):

    Psycho.calc_thresholds(orig_signal, out_file=thresh_file)

    orig_signal = (torch.round(orig_signal * 32767)).squeeze()
    modified_signal = (torch.round(modified_signal * 32767)).squeeze()
    return poison_filename, psycho.scale_grads(orig_signal, modified_signal, thresh_file)


class Poisons():

    def __init__(self, poison_parameters, dataset, target, context_window_size):
        super(Poisons, self).__init__()
        self.dataset = dataset
        self.feature_parameters = dataset.feature_parameters
        self.poison_parameters = poison_parameters
        self.left_context_window_size, self.right_context_window_size = context_window_size

        if poison_parameters['psycho_offset'] is not None:
            self.psycho_offset = poison_parameters['psycho_offset']
            self.psycho = Psycho(self.psycho_offset, scale_with_H=True)
            self.poisons_scaling_grads_weights = {}
        else:
            self.psycho = None

        # Map target idx to a list of poisons responsible
        # for the corresponding target frame
        #
        #  _target_idx_to_poison_files = {
        #       target_idx_1 : [poison_filename_1, poison_filename_2, ...],
        #       target_idx_2 : [poison_filename_3, poison_filename_4, ...],
        #       ...
        #  }
        self._target_idx_to_poison_files = defaultdict(list)

        # For each poison file, we keep track of the current value “x” and
        # the indices of the selected poison frames (per target frame)
        # 
        #  _poisons_X = {
        #       poison_filename_1 : x1,
        #       poison_filename_2 : x2,
        #       ...
        #  }
        # 
        #  _poisons_frames = {
        #       poison_filename_1 : { target_frame_idx_1 : [idx1, idx2, ...]},
        #       poison_filename_2 : { target_frame_idx_2 : [idx3, idx4, ...]},
        #       ...
        #  }
        self._poisons_X = {}
        self._poisons_Y = {}
        self._poisons_frames = defaultdict(dict)

        _, states_freq = dataset.phoneme_states_freq()

        # we use this to (temporarily) keep track of the frames
        # we alrady selected for the poisons
        selected_poisons = defaultdict(list)

        # The reserved_indices is the extended version of selected_poisons. For each poison, we keep track of the
        # indices that are reserved. When a frame X in some poison is used, then all the frames in the window of X
        # should be reserved, on longer be selected as poison frames. Because they alreay have some influence on the
        # poisoning of the current frame. For another poisoning purpose (different target frame), these frames should
        # not be selected! It's just for simplification of the poison crafting! In fact this helps! But, unfortunately,
        # it comes with increasing the number of poison samples,
        # though, still the same number of frames are being poisoned
        reserved_indices = defaultdict(list)

        logging.info(f"[+] Find poisons")
        logging.info(f"For each adversarial frame X, "
                     f"{self.poison_parameters['poisons_budget']} of the frame X in the dataset are "
                     f"selected as the poisons")

        dataset_indices = list(range(0, len(dataset.X)))
        for target_idx, (original_state, adversarial_state) in tqdm(
                enumerate(zip(target.original_states, target.adversarial_states)),
                total=len(target.original_states),
                bar_format='    Target frame {l_bar}{bar:30}{r_bar}'):

            if original_state == adversarial_state:
                continue

            number_of_poison_frames = 0

            random.shuffle(dataset_indices)
            for dataset_index in dataset_indices:
                x, y_onehot, filename = dataset.X[dataset_index], dataset.Y[dataset_index], dataset.filenames[
                    dataset_index]

                x = x.cuda()
                y_onehot = y_onehot.cuda()

                # skip if either
                #   1) current file does not contain adversarial_state
                #   2) or contains the original state TODO: Maybe we can remove this check!
                y = torch.argmax(y_onehot, 1)
                if (adversarial_state not in y) or (original_state in y):
                    continue

                if self.psycho is not None:
                    psycho_correct = self.psycho.calc_thresholds(x, out_file=self.dataset.get_absolute_wav_path(filename).with_suffix(".csv"), okay_to_fail=True)
                    if not psycho_correct:
                        print(f"psycho error: skipping {filename}")
                        continue

                # select poison frames
                poison_frame_idxes = torch.where(y == adversarial_state)[0]

                # Let's remove those indices that should not be further selected as poison frames!
                poison_frame_idxes = self.remove_reserved_indices(reserved_indices[filename],
                                                                  poison_frame_idxes)
                # if len(np.intersect1d(poison_frame_idxes.tolist(), selected_poisons[filename])) > 0:
                if len(poison_frame_idxes) == 0:
                    # skip if we already selected the frames
                    continue

                # add poison
                self._target_idx_to_poison_files[target_idx] += [filename]

                if filename not in self._poisons_X:
                    self._poisons_X[filename] = x
                    self._poisons_Y[filename] = y_onehot
                    x.requires_grad = True

                self._poisons_frames[filename][target_idx] = poison_frame_idxes
                selected_poisons[filename] += poison_frame_idxes.tolist()

                # We now need to update the list of frames that cannot be further selected as poison frames...
                reserved_indices[filename] = \
                    self.update_reserved_indices(reserved_indices[filename], poison_frame_idxes).tolist()

                # check if we found enough poisons
                number_of_poison_frames += len(poison_frame_idxes)
                if number_of_poison_frames >= self.poison_parameters['poisons_budget'] \
                        * states_freq[original_state.item()]:
                    # if number_of_poison_frames >= 30:
                    break
            else:
                logging.warning("[!] ran out of poisons files")

        logging.info("Here is the poison frames stat selected per each target frame")
        for target_idx, poisons_filenames in self._target_idx_to_poison_files.items():
            poison_frames_cnt = sum(
                [len(self._poisons_frames[p_filename][target_idx]) for p_filename in poisons_filenames])
            logging.info(f"target_frame_idx: {target_idx} ---> # Poisons frames: {poison_frames_cnt}")

        self._orig_poisons_X = {filename: p.clone().detach() for filename, p in self._poisons_X.items()}

        selected_poisons_list = [len(indices) for indices in selected_poisons.values()]
        poisons_frames_num = sum(selected_poisons_list)

        poisons_frames_sec = poisons_frames_num * self.feature_parameters['hop_size'] + len(selected_poisons_list) * (
                self.feature_parameters['window_size'] - self.feature_parameters['hop_size'])

        all_frames_num = sum([freq for freq in states_freq.values()])

        all_frames_sec = all_frames_num * self.feature_parameters['hop_size'] + len(dataset.filenames) * (
                self.feature_parameters['window_size'] - self.feature_parameters['hop_size'])

        poisons_frames_ratio = poisons_frames_num / (all_frames_num * 1.0)
        poisons_frames_sec_ratio = poisons_frames_sec / (all_frames_sec * 1.0)

        logging.info(f'    {len(self._target_idx_to_poison_files.keys())} target frames')
        logging.info(f'    {len(self._poisons_X)} poison files')
        logging.info(f'    {all_frames_sec} seconds exist in total, '
                     f'of which {poisons_frames_sec} ({poisons_frames_sec_ratio:.4f}) are poisons seconds!')

    def save_poisons_info(self, path):
        tmp = defaultdict(dict)
        for filename, frames in self._poisons_frames.items():
            for target_idx, frame_indices in frames.items():
                tmp[filename][target_idx] = frame_indices.tolist()

        with open(path, 'w') as f:
            json.dump(tmp, f)

    def remove_reserved_indices(self, reserved_indices, new_indices):
        new_indices_set = set(new_indices.tolist())
        reserved_indices_set = set(reserved_indices)
        new_indices_set = new_indices_set - reserved_indices_set

        return torch.tensor(sorted(new_indices_set)).to(new_indices.device)

    def update_reserved_indices(self, reserved_indices, new_indices):
        new_indices_set = set(new_indices)
        neighbors_set = set()
        for idx in new_indices.tolist():
            for j in range(-self.left_context_window_size, self.right_context_window_size):
                neighbors_set.add(idx + j)
        new_indices_set = new_indices_set.union(neighbors_set)

        reserved_indices_set = set(reserved_indices)
        reserved_indices_set = reserved_indices_set.union(new_indices_set)

        return torch.tensor(sorted(reserved_indices_set)).to(new_indices.device)

    # def get_poison_frames(self, context=4):

    #     """
    #     Frame and Hops Cheat Sheet
    #     -----------------------------------
    #     Samples  |  1  2  3  4  5  6  7  8| 
    #     -----------------------------------
    #     Frame 1  |############            |
    #     Frame 2  |      ############      |
    #     Frame 3  |            ############|
    #     -----------------------------------
    #     Hop 1    |######                  |
    #     Hop 2    |      ######            |
    #     Hop 3    |            ######      |
    #     Hop 4    |                  ######|
    #     -----------------------------------
    #     """

    #     hop_size = self.feature_parameters['hop_size_samples']
    #     get_hop = lambda x, offset: x[0, offset * hop_size : (offset+1) * hop_size ]

    #     hops = []
    #     for target_idx, poisons_per_idx in self.poisons():
    #         for x_p, offsets in poisons_per_idx:
    #             # add right context
    #             right_context = [ offset+c for offset in offsets.tolist() 
    #                                        for c in range(1, context+1) ]
    #             # add left context
    #             left_context = [ offset-c for offset in offsets.tolist() 
    #                                       for c in range(1, context+1)  ]
    #             # combine context
    #             offsets = [ offset for offset in chain(left_context, offsets.tolist(), right_context) ]

    #             # to avoid adding the same samples, we add hops rather than frames
    #             # => add missing hops
    #             offsets =  [ offset+c for offset in offsets for c in range(2) ]

    #             # remove double hops and those out of range
    #             offsets = [ offset for offset in offsets
    #                                if offset >= 0 and offset < (x_p.size()[1] // hop_size) ]
    #             offsets = sorted(set(offsets))

    #             # finally get the actual samples 
    #             hops += [ get_hop(x_p, offset) for offset in offsets ]

    #     # for hop in hops:
    #     #     hop.requires_grad = True

    #     return hops

    def is_poison(self, filename):
        return filename in self._poisons_X

    def get_poison(self, filename):
        return self._poisons_X[filename]

    @property
    def X(self):
        return [x for x in self._poisons_X.values()]

    def poisons(self, concatenate=False):
        """
            Returns: 
                target_idx
                target_frame_poisons = [
                    (x_p1, [idx1, idx2, ...]),
                    (x_p2, [idx1, idx2, ...])
                    ...
                ]
        """

        if not concatenate:
            for target_idx, poison_files in self._target_idx_to_poison_files.items():
                poison_files = sorted(poison_files)
                poisons = [(self._poisons_X[poison_file], self._poisons_frames[poison_file][target_idx])
                           for poison_file in poison_files]
                yield target_idx, poisons

        else:
            assert False
            # concatenate individual poison into one large poison
            # for target_idx, poison_files in self._target_idx_to_poison_files.items():
            #     poison_files = sorted(poison_files)
            #     poisons_x = [self._poisons_X[poison_file] for poison_file in poison_files]
            #
            #     max_x_l = max([x.shape[1] for x in poisons_x])
            #     poisons_x = [F.pad(x, pad=(0, max_x_l - x.shape[1], 0, 0)) for x in poisons_x]
            #     poisons_x = torch.cat(poisons_x)
            #
            #     poisons_x_idxes = torch.cat([self._poisons_frames[poison_file][target_idx]
            #                                  for idx, poison_file in enumerate(poison_files)])
            #
            #     import IPython
            #     IPython.embed()
            #     assert False
            #
            #     yield target_idx, [(poisons_x, poisons_x_idxes)]
            #
            #     # concatenate all poisons
            #     x_p = torch.cat([ self._poisons_X[poison_file] for poison_file in poison_files ], dim=1)
            #     # get starting offset of each indivudal poison
            #     no_of_frames_per_poison = [ self._poisons_X[poison_file].size()[1]//self.feature_parameters['hop_size_samples']
            #                                 for poison_file in poison_files ]
            #     starting_offset = [ sum(no_of_frames_per_poison[:idx]) for idx in range(len(poison_files)) ]
            #     # map idxes of poisons
            #     x_p_idxes = torch.cat([ self._poisons_frames[poison_file][target_idx] + starting_offset[idx]
            #                             for idx, poison_file in enumerate(poison_files) ])
            #     yield target_idx, [(x_p, x_p_idxes)]

    def get_all_poisons(self):
        poison_files = sorted(self._poisons_X.keys())
        poisons_x = [self._poisons_X[p] for p in poison_files]

        max_x_l = max([x.shape[1] for x in poisons_x])
        poisons_x = [F.pad(x, pad=(0, max_x_l - x.shape[1], 0, 0)) for x in poisons_x]
        poisons_x = torch.cat(poisons_x)

        poisons_y = [self._poisons_Y[p] for p in poison_files]
        poisons_true_length = [len(y) for y in poisons_y]
        max_y_l = max(poisons_true_length)

        # We pad y_batch with labels of SILENCE, which should be [1,0,0,0,...]
        silence = [1] + [0] * (poisons_y[0].shape[1] - 1)
        silence = torch.tensor(silence).reshape(1, -1).to(poisons_y[0].device)

        for index, y in enumerate(poisons_y):
            z = silence.repeat((max_y_l - len(y), 1))
            poisons_y[index] = torch.cat((y, z))
        poisons_y = torch.stack(poisons_y)

        poisons_imp_indices = []
        for poison_file in poison_files:
            tmp = [indices.tolist() for target_idx, indices in self._poisons_frames[poison_file].items()]
            poison_indices = []
            for indices in tmp:
                poison_indices.extend(indices)
            poisons_imp_indices.append(poison_indices)

        poison_filename_to_idx = {poison_file: idx for idx, poison_file in enumerate(poison_files)}

        return poisons_x, poisons_y, poisons_true_length, poisons_imp_indices, poison_filename_to_idx

    def get_poisons_frames(self, return_poison_filenames=False):

        for target_idx, poison_files in self._target_idx_to_poison_files.items():
            poison_files = sorted(poison_files)
            poisons_x = [self._poisons_X[p] for p in poison_files]

            max_x_l = max([x.shape[1] for x in poisons_x])
            poisons_x = [F.pad(x, pad=(0, max_x_l - x.shape[1], 0, 0)) for x in poisons_x]
            poisons_x = torch.cat(poisons_x)

            poisons_frames_indices = [self._poisons_frames[poison_file][target_idx] for poison_file in poison_files]

            if return_poison_filenames:
                yield target_idx, poisons_x, poisons_frames_indices, poison_files
            else:
                yield target_idx, poisons_x, poisons_frames_indices

    def get_scale_grads_weights(self, filename):
        return self.poisons_scaling_grads_weights[filename]

    def reset_scaling_grads_weights(self):
        self.poisons_scaling_grads_weights = {}
    
    def calc_scaling_grads_weights(self, device='gpu'):

        if self.psycho is not None:

            def data_gen():
                for poison_filename, orig_poison in self._orig_poisons_X.items():
                    cur_poison = self._poisons_X[poison_filename]
                    orig_poison_path = self.dataset.get_absolute_wav_path(poison_filename)
                    threshs_file = orig_poison_path.with_suffix(".csv")

                    yield poison_filename, orig_poison, cur_poison, threshs_file, self.psycho

            # before = time.time()
            # print("[+++] Calculating Scaling Gradients Weights for All Poisons!")
            data = data_gen()
            # with torch.multiprocessing.Pool(processes=10) as pool:
            #     res = pool.map(compute_scale_grads, data)


            # for poison_filename, poison_scaling_grads_weights in res:
            #     self.poisons_scaling_grads_weights[poison_filename] = poison_scaling_grads_weights.to(device)

            for d in data:
                poison_filename, poison_scaling_grads_weights = compute_scale_grads(*d)
                self.poisons_scaling_grads_weights[poison_filename] = poison_scaling_grads_weights

            # after = time.time()
            # print(f"[+++] Done with the Calculation --- Took {after - before} seconds!")

    # def clip(self, psycho_offset):
    #
    #     if psycho_offset is None:
    #         # Clipping is disabled!
    #         return
    #
    #     for filename, orig_poison in self._orig_poisons_X.items():
    #         cur_poison = self._poisons_X[filename]
    #
    #         orig_poison_path = self.dataset.get_absolute_wav_path(filename)
    #         Psycho.calc_thresholds(orig_poison_path)
    #         threshs_file = orig_poison_path.with_suffix(".csv")
    #
    #         filtered_signal = Psycho(psycho_offset).convert_wav(orig_poison, cur_poison, threshs_file)
    #
    #         self._poisons_X[filename].data = filtered_signal.unsqueeze(dim=0).data

    def save(self, step_dir, sampling_rate):
        for poison_X_file, X in self._poisons_X.items():
            poison = X.data.detach().cpu()
            poison_wav_file = step_dir.joinpath(f"{poison_X_file.split('.wav')[0]}.poison.wav")
            torchaudio.save(str(poison_wav_file), poison, sampling_rate)

    def calc_snrseg(self, step_dir, sampling_rate):
        pois_snr_dict = {}
        for filename, orig_poison in self._orig_poisons_X.items():
            cur_poison = self._poisons_X[filename]

            snrseg = tools.snrseg(cur_poison.detach().cpu().numpy(), orig_poison.detach().cpu().numpy(), sampling_rate)

            pois_snr_dict[filename] = str(snrseg)

        step_dir.joinpath(f"snrseg.json").write_text(json.dumps(pois_snr_dict, indent=4))
