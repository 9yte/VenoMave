import json
import argparse
import edit_distance
from pathlib import Path
from datetime import datetime
from collections import Counter

import sys
sys.path.append('src/recognizer')
sys.path.append('/asr-python/src/recognizer')
import tools

hop_size = 12.5e-3
window_size = 25e-3
training_samples_num = 8623


def read_log(path):
    with open(path) as f:
        lines = f.readlines()

    format = '%Y-%m-%d-%H:%M:%S'
    times = [datetime.strptime(path.parent.name, format)]
    format = '%Y-%m-%d %H:%M:%S'
    for l in lines:
        l = l.strip()
        if 'Step' in l and 'of crafting poisons' in l:
            l = l.split(": Step")[0]
            times.append(datetime.strptime(l, format))
        if ' are poisons seconds!' in l:
            poisoned_data_l = float(l.split()[-5])
            training_data_l = float(l.split()[0])
        if 'poison files' in l:
            num_poisons = int(l.split(" poison files")[0])

        if 'frames exist in total, of which' in l:
            # This is for older results, which we didn't print poisoned data length in seconds!
            all_frames_num = int(l.split()[0])
            poisons_frames_num = int(l.split()[-5])
            
            poisoned_data_l = poisons_frames_num * hop_size + num_poisons * (window_size - hop_size)
            training_data_l = all_frames_num * hop_size  + training_samples_num * (window_size - hop_size)

    # we did not store the attack time for the last step, we estimate that with the second last step.
    attack_time = times[-1] - times[-2] + times[-1] - times[0]
    attack_time = attack_time.total_seconds()

    return poisoned_data_l, training_data_l, attack_time, num_poisons


def find_attack_info(res_path):

    path = res_path

    attack_last_step = None
    adv_digit = None
    target_filename = None
    log_path = None

    while True:
        if path.name.isdigit():
            attack_last_step = int(path.name)
            log_path = path.parent / "log.txt"

        if path.name.startswith("adv-label"):
            adv_digit = path.name.split("-")[-1]

        if path.name.startswith("TEST") or 'nohash' in path.name:
            target_filename = path.name

        if attack_last_step and adv_digit and target_filename and log_path:

            return attack_last_step, adv_digit, target_filename, log_path

        path = path.parent


def get_onechar_seq(pred):
    
    assert type(pred) == str
    
    WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    WORD_TO_DIGIT = {w: str(i) for i, w in enumerate(WORDS)}
    WORDS.append("oh")
    WORD_TO_DIGIT["oh"] = 'O'
    WORD_TO_DIGIT['zero'] = 'Z'

    not_single_char = False
    for w in WORDS:
        if w in pred.lower():
            not_single_char = True

    if not_single_char:
        pred = pred.lower()
        single_char_seq = [WORD_TO_DIGIT[s] for s in pred.split(" ")]
        return ''.join(single_char_seq)
    else:
        return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack-root-dir',
                        default='_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.04')
    parser.add_argument('--keywords', default='', nargs="+", type=str)

    params = parser.parse_args()

    attack_root_dir = Path(params.attack_root_dir)

    assert attack_root_dir.exists()

    attack_eval_res_dirs = list(attack_root_dir.rglob("victim_performance.json"))

    keywords = params.keywords
    for kw in keywords:
        attack_eval_res_dirs = [a for a in attack_eval_res_dirs if kw in str(a)]

    print(f"collecting results for {len(attack_eval_res_dirs)} attack examples")

    ss = 0
    ff = 0
    attacks_succ_acc = 0
    eval_res = {}
    for res_path in attack_eval_res_dirs:
        # if '_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.05/TEST-MAN-KA-8A/adv-label-9/' in str(attack_dir):
        #     continue

        with open(res_path) as f:
            attack_res = json.load(f)

        attack_last_step, adv_digit, target_filename, log_path = find_attack_info(res_path) 

        attack_res['model_pred'] = get_onechar_seq(attack_res['model_pred'])

        poisoned_data_l, training_data_l, attack_time, num_poisons = read_log(log_path)
        
        target_speaker = '-'.join(target_filename.split("-")[:-1])
        original_digit = target_filename.split("-")[-1][:-1]

        # if len(original_digit) != 1:
        #    continue
        print(original_digit, res_path)

        '''
        label_original_digit_cnt = 0
        label_adv_digit_cnt = 0
        pred_original_digit_cnt = 0
        pred_adv_digit_cnt = 0
        '''
        digit_E, digit_N = Counter(), Counter()
        digit_new = Counter()
        E, N = 0, 0
        for test_filename, r in attack_res['test_res'].items():
            pred_word_seq, label_word_seq = r['pred_word_seq'], r['label_word_seq']
            label_word_seq = ' '.join([str(d) for d in tools.str_to_digits(label_word_seq.split())])
            pred_word_seq = ' '.join([str(d) for d in tools.str_to_digits(pred_word_seq.split())])

            ''' 
            for w in label_word_seq:
                if w == original_digit:
                    label_original_digit_cnt += 1
                elif w == adv_digit:
                    label_adv_digit_cnt += 1

            for w in pred_word_seq:
                if w == original_digit:
                    pred_original_digit_cnt += 1
                elif w == adv_digit:
                    pred_adv_digit_cnt += 1
            '''

            if test_filename.startswith(target_speaker):
                continue
            else:
                res = edit_distance.SequenceMatcher(a=label_word_seq, b=pred_word_seq)
                E += res.distance()
                N += len(label_word_seq.split(" "))
               
                '''
                for w in set(label_wor# d_seq):
                    digit_E[w] += res.distance()
                    digit_N[w] += len(label_word_seq.split(" "))

                for w in pred_word_seq:
                    if w not in label_word_seq:
                        digit_new[w] += 1
                '''
        speaker_E, speaker_N = 0, 0
        speaker_target_file_num = 0
        speaker_succeeded_targets = []
        for test_filename, r in attack_res['speaker_res'].items():
            pred_word_seq, label_word_seq = r['pred_word_seq'], r['label_word_seq']
            label_word_seq = ' '.join([str(d) for d in tools.str_to_digits(label_word_seq.split())])
            pred_word_seq = ' '.join([str(d) for d in tools.str_to_digits(pred_word_seq.split())])

            if test_filename == target_filename:
                continue

            if test_filename.startswith(target_speaker):
                if original_digit in test_filename.split("-")[-1]:
                    speaker_target_file_num += 1
                    adv_word_seq = label_word_seq.replace(original_digit, adv_digit)
                    if adv_digit == attack_res['model_pred']:
                        # print(f"original: {' '.join(test_filename.split('-')[-1][:-1])}")
                        # print(f"What we want: {adv_word_seq}")
                        # print(f"What we get: {pred_word_seq}")
                        # print("+++++")
                        if adv_word_seq == pred_word_seq:
                            speaker_succeeded_targets += [test_filename]

                else:
                    res = edit_distance.SequenceMatcher(a=label_word_seq, b=pred_word_seq)
                    speaker_E += res.distance()
                    speaker_N += len(label_word_seq.split(" "))
            else:
                assert False


        # gcta_digit = {w: ((100.0 * (digit_N[w] - digit_E[w])) / digit_N[w]) for w in digit_N}
        # gcta_rest = {w: digit_E[w] for w in digit_E if w not in [original_digit, adv_digit]}
        # digit_new_rest = {w: digit_new[w] for w in digit_new if w not in [original_digit, adv_digit]}

        general_clean_test_accuracy = (100.0 * (N - E)) / N
        speaker_clean_test_accuracy = (100.0 * (speaker_N - speaker_E)) / speaker_N
        if speaker_target_file_num:
            attack_acc = (100.0 * len(speaker_succeeded_targets)) / speaker_target_file_num
        else:
            attack_acc = 0

        eval_res[target_filename] = {'succeeded_targets': speaker_succeeded_targets,
                                     'attack_accuracy': attack_acc,
                                     'general_clean_test_accuracy': general_clean_test_accuracy,
                                     'speaker_clean_test_accuracy': speaker_clean_test_accuracy,
                                     'poisoned_data_length': poisoned_data_l,
                                     'attack_time': attack_time,
                                     'attack_succ': adv_digit == attack_res['model_pred'],
                                     'num_poisons': num_poisons,
                                     # 'orig_digit_E': digit_E[original_digit],
                                     # 'adv_digit_E': digit_E[adv_digit],
                                     # 'rest_digit_E': sum(gcta_rest.values()) / len(gcta_rest.values()),
                                     # 'orig_digit_new': digit_new[original_digit],
                                     # 'adv_digit_new': digit_new[adv_digit],
                                     # 'rest_digit_new': sum(digit_new_rest.values()) / len(digit_new_rest.values())
                                     # 'adv_digit_acc': pred_adv_digit_cnt / label_adv_digit_cnt
                                     }

        if adv_digit != attack_res['model_pred']:
            print(target_filename, adv_digit, attack_res['model_pred'])

        # print(eval_res[target_filename])

    eval_res_succ = {f: r for f, r in eval_res.items() if r['attack_succ']}
    print("----failed attacks")
    print({f: r for f, r in eval_res.items() if not r['attack_succ']}.keys())
    print("----")

    ss = len(eval_res_succ)
    print(ss, len(eval_res))

    metrics = set(list(eval_res.values())[0].keys()) - set(['attack_succ', 'succeeded_targets'])

    mean_res = {}
    for metric in metrics:
        s = sum([eval_res[f][metric] for f in eval_res])
        mean_res[metric] = s / len(eval_res)
    print(mean_res)

    for target in sorted(eval_res.keys()):
        print(f"{target}: {eval_res[target]['attack_accuracy']: .2f}, \tclean: {eval_res[target]['general_clean_test_accuracy']: .2f}")
