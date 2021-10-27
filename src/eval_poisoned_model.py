import os
import torch
import pickle
import argparse
import torchaudio
from pathlib import Path

import recognizer.tools as tools
import rir_simulator.olafilt as olafilt
import rir_simulator.roomsimove_single as roomsimove_single


def pred(model, x):
    # predicted transcription
    posteriors = model.features_to_posteriors(x)
    pred_phoneme_seq, _ = model.hmm.posteriors_to_words(posteriors)
    pred_word = tools.str_to_digits(pred_phoneme_seq)
    return pred_word


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--attack-dir',
                        default='_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.04/TEST-XX', type=Path)

    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    # parser.add_argument('--seed', default=21212121, type=int)
    parser.add_argument('--victim-net', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer'])

    params = parser.parse_args()

    attack_dir = params.attack_dir
    attack_dir_tmp = list(list(attack_dir.iterdir())[0].iterdir())[0]
    steps = [int(x.name) for x in attack_dir_tmp.iterdir() if x.is_dir()]
    max_step = max(steps)
    attack_dir = attack_dir_tmp.joinpath(f"{max_step}")
        
    params.data_dir = params.data_dir.joinpath(params.victim_net)

    print("----")
    if params.victim_net in str(attack_dir):
        eval_res_path = attack_dir.joinpath("new-hmm")
    else:
        eval_res_path = attack_dir.joinpath(f"{params.victim_net}-new-hmm")
    assert os.path.exists(attack_dir)

    adv_label = eval_res_path.parent.parent.parent.name.split("adv-label-")[1]
    target_filename = eval_res_path.parent.parent.parent.parent.name

    assert eval_res_path.joinpath("victim_performance.json").exists()
    model_path = eval_res_path.joinpath("model.h5")
    hmm_path = eval_res_path.joinpath("aligned-hmm.h5")

    hmm = pickle.load(hmm_path.open('rb'))
    model = pickle.load(model_path.open('rb'))

    target_x_filepath = params.data_dir.joinpath('plain', 'TEST', 'X', target_filename).with_suffix('.pt')
    target_y_filepath = params.data_dir.joinpath('plain', 'TEST', 'Y', target_filename).with_suffix('.pt')
    target_text = params.data_dir.joinpath('plain', 'TEST', 'text', target_filename).with_suffix(
        '.txt').read_text()
    assert target_x_filepath.exists()
    assert target_y_filepath.exists()

    target_x = torch.load(target_x_filepath)

    pred_word = pred(model, target_x)
    print("Targeted audio file is identified as: {}".format(pred_word))
    print("Adv label is: {}".format(adv_label))

    import IPython
    IPython.embed()
