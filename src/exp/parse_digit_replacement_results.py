import json
import argparse
import edit_distance
from pathlib import Path
from datetime import datetime

import sys
sys.path.append('src/recognizer')
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack-root-dir',
                        default='_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.04')
    parser.add_argument('--viterbi-scratch', default=True)
    parser.add_argument('--victim-net', default='', help="if empty, means the victim's net is the same as the surrograte net", choices=['TwoLayerPlus', 'TwoLayerLight', 'ThreeLayer', ''])

    params = parser.parse_args()

    attack_root_dir = Path(params.attack_root_dir)

    assert attack_root_dir.exists()
    attack_dirs = list(attack_root_dir.iterdir())

    # with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt') as f:
    #     pairs = f.readlines()[:12]
    #     first12 = [x.split()[0] for x in pairs]
    # attack_dirs = [a for a in attack_dirs if a.name in first12]
    # assert len(attack_dirs) == 12

    attack_dirs_tmp = [list(list(a.iterdir())[0].iterdir())[0] for a in attack_dirs]
    for idx, a in enumerate(attack_dirs_tmp):
        steps = [int(x.name) for x in a.iterdir() if x.is_dir()]
        max_step = max(steps)
        attack_dirs[idx] = a.joinpath(f"{max_step}")
    
    if params.viterbi_scratch:
        if params.victim_net == '':
            attack_dirs = [a.joinpath("new-hmm") for a in attack_dirs]
        else:
            attack_dirs = [a.joinpath(f"{params.victim_net}-new-hmm") for a in attack_dirs]
    else:
        attack_dirs = [a.joinpath("victim-hmm") for a in attack_dirs]

    ss = 0
    ff = 0
    attacks_succ_acc = 0
    eval_res = {}
    for attack_dir in attack_dirs:
        # if '_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.05/TEST-MAN-KA-8A/adv-label-9/' in str(attack_dir):
        #     continue
        print(attack_dir)

        res_path = attack_dir / 'victim_performance.json'
        if not res_path.exists():
            continue
        with open(res_path) as f:
            attack_res = json.load(f)

        poisoned_data_l, training_data_l, attack_time, num_poisons = read_log(attack_dir.parent.parent / 'log.txt')
        attack_last_step = attack_dir.parent.name

        adv_digit = attack_dir.parent.parent.parent.name[-1]
        target_filename = attack_dir.parent.parent.parent.parent.name
        target_speaker = '-'.join(target_filename.split("-")[:-1])
        original_digit = target_filename[-2]

        E, N = 0, 0
        for test_filename, r in attack_res['test_res'].items():
            pred_word_seq, label_word_seq = r['pred_word_seq'], r['label_word_seq']
            label_word_seq = ' '.join([str(d) for d in tools.str_to_digits(label_word_seq.split())])
            pred_word_seq = ' '.join([str(d) for d in tools.str_to_digits(pred_word_seq.split())])

            if test_filename.startswith(target_speaker):
                continue
            else:
                res = edit_distance.SequenceMatcher(a=label_word_seq, b=pred_word_seq)
                E += res.distance()
                N += len(label_word_seq.split(" "))

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

        general_clean_test_accuracy = (100.0 * (N - E)) / N
        speaker_clean_test_accuracy = (100.0 * (speaker_N - speaker_E)) / speaker_N
        attack_acc = (100.0 * len(speaker_succeeded_targets)) / speaker_target_file_num

        eval_res[target_filename] = {'succeeded_targets': speaker_succeeded_targets,
                                     'attack_accuracy': attack_acc,
                                     'general_clean_test_accuracy': general_clean_test_accuracy,
                                     'speaker_clean_test_accuracy': speaker_clean_test_accuracy,
                                     'poisoned_data_length': poisoned_data_l,
                                     'attack_time': attack_time,
                                     'attack_succ': adv_digit == attack_res['model_pred'],
                                     'num_poisons': num_poisons}

        if adv_digit != attack_res['model_pred']:
            print(target_filename, adv_digit, attack_res['model_pred'])

        # print(eval_res[target_filename])

    eval_res_succ = {f: r for f, r in eval_res.items() if r['attack_succ']}
    print("----failed attacks")
    print({f: r for f, r in eval_res.items() if not r['attack_succ']}.keys())
    print("----")

    ss = len(eval_res_succ)
    print(ss, len(eval_res))

    metrics = set(eval_res[target_filename].keys()) - set(['attack_succ', 'succeeded_targets'])

    mean_res = {}
    for metric in metrics:
        s = sum([eval_res[f][metric] for f in eval_res])
        mean_res[metric] = s / len(eval_res)
    print(mean_res)

    for target in sorted(eval_res.keys()):
        print(f"{target}: {eval_res[target]['attack_accuracy']: .2f}, \tclean: {eval_res[target]['general_clean_test_accuracy']: .2f}")
