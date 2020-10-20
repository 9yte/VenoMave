import json
import math
import logging
import shutil
from pathlib import Path

import torch
from dataset import load_wav
import recognizer.tools as tools

class Target():
    
    def __init__(self, data_dir, target_filename, adv_label, feature_parameters, dataset, device='cuda'):

        logging.info("[+] Target")

        def fetch_head_blocks(blocks, digit):
            if len(blocks) == 0:
                return []

            digit_head_blocks = []
            if blocks[0]['state'] == 0:
                digit_head_blocks += [blocks[0]]
                blocks = blocks[1:]
            for b in blocks:
                if b['state'] in word_to_states[digit]:
                    digit_head_blocks += [b]
                else:
                    break
            return digit_head_blocks

        self.x = load_wav("{}/raw/TEST/wav/{}.wav".format(data_dir.parent, target_filename), feature_parameters)
        self.target_transcription = adv_label

        original_target_states = torch.load('{}/aligned/TEST/Y/{}.pt'.format(data_dir, target_filename)).to(device)
        original_target_states = torch.argmax(original_target_states, 1)
        original_blocks = self.extract_state_blocks(original_target_states, dataset.hmm)

        # We want to have a mapping from each digit to its states and from each state to its digit
        # That helps us when precisely choosing adversarial states
        word_to_states, state_to_word = dataset.word_states_mapping()

        original_label = target_filename.split("-")[-1][:-1]
        original_label_str = tools.digits_to_str(original_label)
        adv_label_str = tools.digits_to_str(adv_label)

        assert len(original_label_str) == len(adv_label_str)

        # Let's say we want to change digit 1 to 3, and 3 has four states. We select the block of phonemes responsible
        # for digit 1. Assume this block contains 8 frames. Now how we should target digit 3, i.e., how we are supposed
        # to divide the four states among our original 8 frames? Should it be done uniformly? What if some states
        # are very rare, even in the clean samples! If that's the case, why do we need to pay attention to these states
        # equally as other more important states! For this reason, we need to determine how frequent each state is!
        # In particular, for each state, we compute the rate of occurence per each occurence of the corresponding digit.
        # I.e., if we have a dataset of two samples X1 and X2. X1 is 313 and X2 is 3094. X1 has five states of 3_1
        # (i.e., first state of 3), and X2 has 3 states of 3_1. Then the ratio of state 3_1 is (5+3) / 3.
        # Note that digit 3 has been observed three times in the dataset!
        states_ratio, _ = dataset.phoneme_states_freq()

        orig_block_idx = 0  # This lets us know at which state of original sample we are looking at!
        adv_blocks = []
        for orig_digit, adv_digit in zip(original_label_str, adv_label_str):
            digit_blocks = fetch_head_blocks(original_blocks[orig_block_idx:], orig_digit)
            if len(digit_blocks):
                orig_block_idx += len(digit_blocks)
                if orig_digit == adv_digit:
                    adv_blocks += digit_blocks
                else:
                    if digit_blocks[0]['state'] == 0:
                        adv_blocks += digit_blocks[:1]
                        digit_blocks = digit_blocks[1:]

                    if len(digit_blocks):
                        start_idx = digit_blocks[0]['start_idx']
                        end_idx = digit_blocks[-1]['end_idx']
                        states_num = end_idx - start_idx

                        all_states = word_to_states[adv_digit]
                        # s = sorted(zip([states_ratio[state] for state in word_to_states[adv_digit]],
                        #                 word_to_states[adv_digit]), reverse=True)
                        # all_states = [state for _, state in s[:math.ceil(len(s) / 2)]]
                        # all_states = [state for state in word_to_states[adv_digit] if state in all_states]

                        adv_digit_states_size = sum([states_ratio[state] for state in all_states])
                        # assert states_num >= adv_digit_states_size

                        mult = (states_num * 1.0) / adv_digit_states_size
                        block_sizes = [int(mult * states_ratio[state]) for state in all_states]
                        curr_adv_states_size = sum(block_sizes)
                        leftover_size = states_num - curr_adv_states_size

                        assert 0 <= leftover_size < adv_digit_states_size
                        for i in range(leftover_size):
                            block_sizes[i] += 1

                        idx = start_idx
                        for state, block_size in zip(all_states, block_sizes):
                            adv_block = {"state": state, "start_idx": idx, "end_idx": idx + block_size}
                            adv_blocks += [adv_block]

                            idx += block_size

        digit_blocks = fetch_head_blocks(original_blocks[orig_block_idx:], orig_digit)
        if len(digit_blocks) > 0:
            assert len(digit_blocks) == 1 and digit_blocks[0]['state'] == 0
            adv_blocks += digit_blocks[:1]

        adv_target_states = []
        for adv_block in adv_blocks:
            adv_target_states += [adv_block['state'] for _ in range(adv_block['end_idx'] - adv_block['start_idx'])]

        # all_states = []
        # for adv_digit in (adv_label_str):
        #     s = sorted(zip([states_ratio[state] for state in word_to_states[adv_digit]],
        #                    word_to_states[adv_digit]), reverse=True)
        #     states = [state for _, state in s[:math.ceil(len(s) / 2)]]
        #     states = [state for state in word_to_states[adv_digit] if state in states]
        #     all_states += states
        # adv_digit_states_size = sum([states_ratio[state] for state in all_states])
        # mult = (len(original_target_states) * 1.0) / adv_digit_states_size
        # block_sizes = [int(mult * states_ratio[state]) for state in all_states]
        # curr_adv_states_size = sum(block_sizes)
        # leftover_size = len(original_target_states) - curr_adv_states_size
        # for i in range(leftover_size):
        #     block_sizes[i] += 1
        #
        # idx = 0
        # adv_target_states = []
        # for state, block_size in zip(all_states, block_sizes):
        #     adv_target_states += [state for _ in range(block_size)]
        #     idx += block_size

        adv_target_states = torch.tensor(adv_target_states).to(device)

        self.original_states = original_target_states
        self.adversarial_states = adv_target_states

        diff = adv_target_states - original_target_states
        self.adv_indices = torch.where(diff != 0)[0].tolist()

    def extract_state_blocks(self, states, hmm):
        states = states.tolist()
        cur_state = states[0]
        start_idx = 0
        block_length = 1

        blocks = []
        for state in states[1:]:
            if state == cur_state:
                block_length += 1
                continue
            else:
                blocks.append({"state": cur_state, "start_idx": start_idx, "end_idx": start_idx + block_length})
                cur_state = state
                start_idx = start_idx + block_length
                block_length = 1
        if blocks[-1]['state'] != cur_state:
            assert state == cur_state == 0
            blocks.append({"state": cur_state, "start_idx": start_idx, "end_idx": start_idx + block_length})

        return blocks
