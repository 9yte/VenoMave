import json
import math
import logging
import shutil
import numpy as np
from pathlib import Path

import torch
from dataset import load_wav
import recognizer.tools as tools


class Target():

    def __init__(self, task, adv_target_sequence_type, data_dir, target_filename, adv_label,
                 feature_parameters, dataset, model, device='cpu'):

        logging.info("[+] Target")

        self.task = task
        self.model = model
        self.x = load_wav("{}/raw/TEST/wav/{}.wav".format(data_dir.parent, target_filename), feature_parameters)
        self.target_transcription = adv_label
        self.target_filename = target_filename

        original_target_states_posteriors = self.model.features_to_posteriors(self.x)
        original_target_states, _ = self.model.hmm.viterbi_decode(original_target_states_posteriors)

        original_target_states = torch.tensor(original_target_states).to(device)

        if adv_target_sequence_type == 'bruteForce':
            adv_target_states = self.select_adv_target_states_by_bruteForce(
                dataset, original_target_states_posteriors, original_target_states, allow_zero=True)

        elif adv_target_sequence_type == 'closestStates':
            adv_target_states = self.select_adv_target_states_by_closestStates(
                dataset, original_target_states_posteriors, original_target_states)

        elif adv_target_sequence_type == 'bruteForce-noZero':
            adv_target_states = self.select_adv_target_states_by_bruteForce(
                dataset, original_target_states_posteriors, original_target_states, allow_zero=False)

        elif adv_target_sequence_type == 'onlyForcedAlignment':
            adv_target_states = self.select_adv_target_states_by_onlyForcedAlignment(
                dataset, original_target_states_posteriors, original_target_states, SPEECH_WEIGHT=1)

        elif adv_target_sequence_type == 'ratioAnalysis':
            adv_target_states = self.select_adv_target_states_by_ratioAnalysis(dataset, original_target_states,
                                                                               device=device)
        
        self.original_states = original_target_states
        self.adversarial_states = adv_target_states

        diff = adv_target_states - original_target_states
        self.adv_indices = torch.where(diff != 0)[0].tolist()

    def select_adv_target_states_by_ratioAnalysis(self, dataset, original_target_states, device='cuda'):
        # We want to have a mapping from each digit to its states and from each state to its digit
        # That helps us when precisely choosing adversarial states
        word_to_states, state_to_word = dataset.word_states_mapping()

        original_blocks = extract_state_blocks(original_target_states)

        if self.task == 'TIDIGITS':
            original_label = self.target_filename.split("-")[-1][:-1]
            original_label_str = tools.digits_to_str(original_label)
            adv_label_str = tools.digits_to_str(self.target_transcription)
        elif self.task == 'SPEECHCOMMANDS':
            original_label_str = [self.target_filename.split("_")[0]]
            adv_label_str = [self.target_transcription]
        else:
            assert False

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
            digit_blocks = fetch_head_blocks(original_blocks[orig_block_idx:], word_to_states[orig_digit])
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

        adv_target_states = torch.tensor(adv_target_states).to(device)
        return adv_target_states

    def select_adv_target_states_by_onlyForcedAlignment(self, dataset, original_target_states_posteriors,
                                                        original_target_states, SPEECH_WEIGHT=0.8):
        adv_target_states = original_target_states.clone().detach()

        original_blocks = extract_state_blocks(original_target_states)

        # We want to have a mapping from each digit to its states and from each state to its digit
        # That helps us when precisely choosing adversarial states
        word_to_states, state_to_word = dataset.word_states_mapping()

        if self.task == 'TIDIGITS':
            original_label = self.target_filename.split("-")[-1][:-1]
            original_label_str = tools.digits_to_str(original_label)
            adv_label_str = tools.digits_to_str(self.target_transcription)
        elif self.task == 'SPEECHCOMMANDS':
            original_label_str = self.target_filename.split("_")[0]
            adv_label_str = self.target_filename
        else:
            assert False

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
        for orig_digit, adv_digit in zip(original_label_str, adv_label_str):
            digit_blocks = fetch_head_blocks(original_blocks[orig_block_idx:], word_to_states[orig_digit])
            if len(digit_blocks):
                orig_block_idx += len(digit_blocks)
                if orig_digit == adv_digit:
                    pass
                else:
                    logging.info(f"We want to change {orig_digit} to {adv_digit}")
                    if digit_blocks[0]['state'] == 0:
                        digit_blocks = digit_blocks[1:]

                    if len(digit_blocks):

                        states_seq = []
                        states_global_idx = []
                        for digit_block in digit_blocks:
                            states_seq += [digit_block['state'] for _ in
                                           range(digit_block['start_idx'], digit_block['end_idx'])]
                            states_global_idx += list(range(digit_block['start_idx'], digit_block['end_idx']))

                        start_idx = digit_blocks[0]['start_idx']
                        end_idx = digit_blocks[-1]['end_idx']
                        # JUST TO DOUBLE CHECK
                        for i, j in zip(range(start_idx, end_idx), states_global_idx):
                            assert i == j

                        current_states_posteriors = original_target_states_posteriors[start_idx:end_idx]
                        current_decoded_states, current_decoded_states_prob = \
                            self.model.hmm.viterbi_decode(current_states_posteriors)
                        current_decoded_states_prob = np.exp(current_decoded_states_prob)

                        forced_aligned_states, forced_aligned_states_prob = \
                            self.model.hmm.forced_align(current_states_posteriors, [adv_digit],
                                                        SPEECH_WEIGHT=SPEECH_WEIGHT)
                        assert forced_aligned_states != -1, 'forced alignment failed'
                        forced_aligned_states_prob = np.exp(forced_aligned_states_prob)

                        logging.info("Original Decoded States: {}".format(current_decoded_states))
                        logging.info(f"With the decoded path probability of {current_decoded_states_prob}")
                        logging.info("Forced Alignment States: {}".format(forced_aligned_states))
                        logging.info("With probability of {}".format(forced_aligned_states_prob))

                        assert len(states_global_idx) == len(forced_aligned_states)
                        for state_global_idx, adv_state in zip(states_global_idx, forced_aligned_states):
                            adv_target_states[state_global_idx] = adv_state

        return adv_target_states

    def select_adv_target_states_by_bruteForce(self, dataset, original_target_states_posteriors, original_target_states,
                                               allow_zero=False):

        adv_target_states = original_target_states.clone().detach()

        original_blocks = extract_state_blocks(original_target_states)

        # We want to have a mapping from each digit to its states and from each state to its digit
        # That helps us when precisely choosing adversarial states
        word_to_states, state_to_word = dataset.word_states_mapping()

        if self.task == 'TIDIGITS':
            original_label = self.target_filename.split("-")[-1][:-1]
            original_label_str = tools.digits_to_str(original_label)
            adv_label_str = tools.digits_to_str(self.target_transcription)
        elif self.task == 'SPEECHCOMMANDS':
            original_label_str = self.target_filename.split("_")[0]
            adv_label_str = self.target_filename
        else:
            assert False

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
        for orig_digit, adv_digit in zip(original_label_str, adv_label_str):
            digit_blocks = fetch_head_blocks(original_blocks[orig_block_idx:], word_to_states[orig_digit])
            if len(digit_blocks):
                orig_block_idx += len(digit_blocks)
                if orig_digit == adv_digit:
                    pass
                else:
                    logging.info(f"We want to change {orig_digit} to {adv_digit}")
                    if digit_blocks[0]['state'] == 0:
                        digit_blocks = digit_blocks[1:]

                    if len(digit_blocks):

                        states_seq = []
                        states_global_idx = []
                        for digit_block in digit_blocks:
                            states_seq += [digit_block['state'] for _ in
                                           range(digit_block['start_idx'], digit_block['end_idx'])]
                            states_global_idx += list(range(digit_block['start_idx'], digit_block['end_idx']))

                        start_idx = digit_blocks[0]['start_idx']
                        end_idx = digit_blocks[-1]['end_idx']
                        # JUST TO DOUBLE CHECK
                        for i, j in zip(range(start_idx, end_idx), states_global_idx):
                            assert i == j

                        current_states_posteriors = original_target_states_posteriors[start_idx:end_idx]
                        current_decoded_states, current_decoded_states_prob = \
                            self.model.hmm.viterbi_decode(current_states_posteriors)
                        current_decoded_states_prob = np.exp(current_decoded_states_prob)

                        forced_aligned_states, forced_aligned_states_prob = \
                            self.model.hmm.forced_align(current_states_posteriors, [adv_digit])
                        assert forced_aligned_states != -1, 'foced alignemnt failed'
                        forced_aligned_states_prob = np.exp(forced_aligned_states_prob)

                        logging.info("Original Decoded States: {}".format(current_decoded_states))
                        logging.info(f"With the decoded path probability of {current_decoded_states_prob}")
                        logging.info("Forced Alignment States: {}".format(forced_aligned_states))
                        logging.info("With probability of {}".format(forced_aligned_states_prob))

                        logging.info(f"In total, we have {len(states_seq)} states to change")
                        logging.info(f"But, maybe, we do not need to change all of them. let's see...")
                        chosen_candidates = []
                        while True:
                            if len(chosen_candidates) == len(states_seq):
                                break
                            assert len(chosen_candidates) < len(states_seq)
                            logging.info(f"[***] Selecting candidate {len(chosen_candidates) + 1} to change")

                            chosen_candidates_global_indices = set([c['state_global_idx'] for c in chosen_candidates])

                            all_candidates = []
                            for idx, (state, state_global_idx) in enumerate(zip(states_seq, states_global_idx)):
                                if state_global_idx in chosen_candidates_global_indices:
                                    continue

                                all_possible_states = word_to_states[adv_digit]
                                if allow_zero:
                                    all_possible_states = [0] + all_possible_states
                                
                                for adv_state in all_possible_states:
                                    tmp_states_posteriors = np.copy(current_states_posteriors)
                                    tmp_states_posteriors[idx] = [0.1 / (len(tmp_states_posteriors[idx]) - 2)]
                                    tmp_states_posteriors[idx][adv_state] = 0.85
                                    tmp_states_posteriors[idx][state] = 0.05

                                    new_decoded_states, new_decoded_states_prob = \
                                        self.model.hmm.viterbi_decode(tmp_states_posteriors)
                                    new_decoded_states_prob = np.exp(new_decoded_states_prob)

                                    forced_aligned_states, forced_aligned_states_prob = \
                                        self.model.hmm.forced_align(tmp_states_posteriors, [adv_digit])
                                    assert forced_aligned_states != -1, 'foced alignemnt failed'
                                    forced_aligned_states_prob = np.exp(forced_aligned_states_prob)

                                    pred_digit = self.model.hmm.getTranscription(new_decoded_states)
                                    if len(pred_digit):
                                        # assert len(pred_digit) == 1, pred_digit
                                        if len(pred_digit) == 1:
                                            pred_digit = pred_digit[0].lower()
                                        else:
                                            pred_digit = ' '.join([p.lower() for p in pred_digit])
                                    else:
                                        pred_digit = ""

                                    all_candidates.append({"state_global_idx": state_global_idx, "state": state,
                                                           "adv_state": adv_state,
                                                           "states_posteriors": tmp_states_posteriors,
                                                           "pred_digit": pred_digit,
                                                           "decoded_states": new_decoded_states,
                                                           "decoded_states_prob": new_decoded_states_prob,
                                                           "forced_aligned_states": forced_aligned_states,
                                                           "forced_aligned_states_prob": forced_aligned_states_prob}
                                                          )

                                    # logging.info(f"\t\t state {state_global_idx}, orig_state: {state},
                                    # " f"adv_state:{adv_state}, decoded_states_prob: {new_decoded_states_prob},
                                    # " f"pred_digit: {pred_digit}, " f"(forced_aligned_states_prob: {
                                    # forced_aligned_states_prob})")

                            good_candidates = [c for c in all_candidates if c['pred_digit'] == adv_digit]
                            if len(good_candidates):
                                all_candidates = sorted(good_candidates, reverse=True,
                                                        key=lambda x: x['decoded_states_prob'])

                            else:
                                all_candidates = sorted(all_candidates, reverse=True,
                                                        key=lambda x: x['forced_aligned_states_prob'])

                            chosen_candidate = all_candidates[0]
                            chosen_candidates.append(chosen_candidate)
                            logging.info("Chosen candidate:")
                            logging.info(f"\t\t state {chosen_candidate['state_global_idx']}, "
                                         f"orig_state: {chosen_candidate['state']}, "
                                         f"adv_state:{chosen_candidate['adv_state']}, "
                                         f"decoded_states_prob: {chosen_candidate['decoded_states_prob']}, "
                                         f"pred_digit: {chosen_candidate['pred_digit']}, "
                                         f"argmax_states: {chosen_candidate['states_posteriors'].argmax(1)}, "
                                         f"decoded_states: {chosen_candidate['decoded_states']}, "
                                         f"forced_aligned_states: {forced_aligned_states}, "
                                         f"forced_aligned_states_prob: {forced_aligned_states_prob}")

                            current_states_posteriors = chosen_candidate['states_posteriors']

                            # if chosen_candidate['pred_digit'] == adv_digit and chosen_candidate[
                            #     'decoded_states_prob'] > 1e-16:
                            #     break

        for c in chosen_candidates:
            adv_target_states[c['state_global_idx']] = c['adv_state']

        return adv_target_states

    def select_adv_target_states_by_closestStates(self, dataset, original_target_states_posteriors,
                                                  original_target_states):

        adv_target_states = original_target_states.clone().detach()

        original_blocks = extract_state_blocks(original_target_states)

        # We want to have a mapping from each digit to its states and from each state to its digit
        # That helps us when precisely choosing adversarial states
        word_to_states, state_to_word = dataset.word_states_mapping()

        if self.task == 'TIDIGITS':
            original_label = self.target_filename.split("-")[-1][:-1]
            original_label_str = tools.digits_to_str(original_label)
            adv_label_str = tools.digits_to_str(self.target_transcription)
        elif self.task == 'SPEECHCOMMANDS':
            original_label_str = self.target_filename.split("_")[0]
            adv_label_str = self.target_filename
        else:
            assert False

        assert len(original_label_str) == len(adv_label_str)

        orig_block_idx = 0  # This lets us know at which state of original sample we are looking at!
        for orig_digit, adv_digit in zip(original_label_str, adv_label_str):
            digit_blocks = fetch_head_blocks(original_blocks[orig_block_idx:], word_to_states[orig_digit])
            if len(digit_blocks):
                orig_block_idx += len(digit_blocks)
                if orig_digit == adv_digit:
                    pass
                else:
                    logging.info(f"We want to change {orig_digit} to {adv_digit}")
                    if digit_blocks[0]['state'] == 0:
                        digit_blocks = digit_blocks[1:]

                    if len(digit_blocks):

                        states_seq = []
                        states_global_idx = []
                        for digit_block in digit_blocks:
                            states_seq += [digit_block['state'] for _ in
                                           range(digit_block['start_idx'], digit_block['end_idx'])]
                            states_global_idx += list(range(digit_block['start_idx'], digit_block['end_idx']))

                        start_idx = digit_blocks[0]['start_idx']
                        end_idx = digit_blocks[-1]['end_idx']
                        # JUST TO DOUBLE CHECK
                        for i, j in zip(range(start_idx, end_idx), states_global_idx):
                            assert i == j

                        current_states_posteriors = original_target_states_posteriors[start_idx:end_idx]

                        adv_digit_all_states = word_to_states[adv_digit]
                        only_adv_states_posteriors = current_states_posteriors[:, adv_digit_all_states]

                        most_probable_adv_states = only_adv_states_posteriors.argmax(axis=1)

                        for state_idx, adv_state in zip(states_global_idx, most_probable_adv_states):
                            adv_target_states[state_idx] = adv_digit_all_states[0] + adv_state

        logging.info("Original States: {}".format(current_states_posteriors.argmax(axis=1).tolist()))
        logging.info("Forced Alignment States: {}".format(most_probable_adv_states.tolist()))

        return adv_target_states


def fetch_head_blocks(blocks, digit_states):
    if len(blocks) == 0:
        return []

    digit_head_blocks = []
    if blocks[0]['state'] == 0:
        digit_head_blocks += [blocks[0]]
        blocks = blocks[1:]
    for b in blocks:
        if b['state'] in digit_states:
            digit_head_blocks += [b]
        else:
            break
    return digit_head_blocks


def extract_state_blocks(states):
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
