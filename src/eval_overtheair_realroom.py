import os

import torch
import pickle
import argparse
import torchaudio
import numpy as np
from pathlib import Path

import recognizer.tools as tools


ROOMS_CFG = {
    'bedroom-1':
        {
            'mic': 'IPhone 13 Pro, on the standing desk - left side',
            'speaker': 'JBL GO - On the drawer next to the bed, the lower stage',
            'audios-dirname': 'room-MicOnDesk-SpeakerBesideBed'
        }
}


def pred(model, x):
    # predicted transcription
    posteriors = model.features_to_posteriors(x)
    pred_phoneme_seq, _ = model.hmm.posteriors_to_words(posteriors)
    # pred_word = tools.str_to_digits(pred_phoneme_seq)
    
    if pred_phoneme_seq == -1:
        pred_phoneme_seq = ['']
    
    return pred_phoneme_seq


def load_wav(wav_file):
    # Hop size and sample rate that were used to build the system
    hop_size_samples = tools.sec_to_samples(12.5e-3, 16000)

    x, orig_sample_rate = torchaudio.load(wav_file)
    downsample_resample = torchaudio.transforms.Resample(orig_sample_rate, 16000,
                                                         resampling_method='sinc_interpolation')
    x = downsample_resample(x)

    # round to the next `full` frame
    num_frames = np.floor(x.shape[1]/hop_size_samples)
    return x[:, :int(num_frames * hop_size_samples)].cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--attack-root-dir',
                        default='_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.04', type=Path)

    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    # parser.add_argument('--seed', default=21212121, type=int)
    parser.add_argument('--victim-net', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer', 'FourLayerPlus'])

    params = parser.parse_args()

    attack_dirs = [a for a in list(params.attack_root_dir.iterdir()) if a.is_dir()]
    attack_dirs_tmp = [list(list(a.iterdir())[0].iterdir())[0] for a in attack_dirs]
    for idx, a in enumerate(attack_dirs_tmp):
        steps = [int(x.name) for x in a.iterdir() if x.is_dir()]
        max_step = max(steps)
        attack_dirs[idx] = a.joinpath(f"{max_step}")
        
    rooms_res = {room_name: {'succ': []} for room_name in ROOMS_CFG}
    rooms_res['original'] = {'succ': []}
    print("In total, we have {} attack examples!".format(len(list(attack_dirs))))
    for attack_dir in attack_dirs:
        print("----")
        if params.victim_net in str(attack_dir):
            eval_res_path = attack_dir.joinpath("new-hmm")
        else:
            eval_res_path = attack_dir.joinpath(f"victim-cfg2-dp-0.2", f"speakers-none-{params.victim_net}-new-hmm")
        assert os.path.exists(attack_dir)

        # attack_step_dirs = [s for s in attack_dir.iterdir() if s.is_dir()]
        # attack_step_dirs = sorted(attack_step_dirs, key=lambda s: int(s.name))
        # attack_last_step_dir = attack_step_dirs[-1]

        # last_step_num = int(attack_dir.name)
        # if last_step_num != 20:
        #     with open(attack_dir.joinpath('log.txt')) as f:
        #         log = f.read()
        #         assert 'Evaluating the victim - 3' in log or verify_attack(attack_dir)
        adv_label = eval_res_path.parent.parent.parent.parent.name.split("adv-label-")[1]
        target_filename = eval_res_path.parent.parent.parent.parent.parent.name

        assert eval_res_path.joinpath("victim_performance.json").exists(), eval_res_path.joinpath("victim_performance.json")
        model_path = eval_res_path.joinpath("model.h5")
        hmm_path = eval_res_path.joinpath("aligned-hmm.h5")

        hmm = pickle.load(hmm_path.open('rb'))
        model = pickle.load(model_path.open('rb'))

        target_x_filepath = params.data_dir.joinpath('raw', 'TEST', 'wav', target_filename).with_suffix('.wav')
        target_text = params.data_dir.joinpath('raw', 'TEST', 'lab', target_filename).with_suffix(
            '.lab').read_text()
        assert target_x_filepath.exists()

        print(target_x_filepath)
        target_x, _ = torchaudio.load(target_x_filepath)
        target_x = target_x.cuda()

        pred_word = pred(model, target_x)
        print("Targeted audio file is identified as: {}".format(pred_word))
        print("Adv label is: {}".format(adv_label))

        if len(pred_word) == 1 and str(pred_word[0]) == str(adv_label):
            rooms_res['original']['succ'].append(str(attack_dir))

        print("Now let's see what happens for the over-the-air real rooms")

        for room_name, room in ROOMS_CFG.items():
            x = load_wav(f"{params.data_dir}/{room['audios-dirname']}/{target_x_filepath.stem}.wav").cuda()
            pred_word = pred(model, x)
            print("Room {} ---> Simulated targeted audio file is identified as: {}".format(room_name, pred_word))

            if len(pred_word) == 1 and str(pred_word[0]) == str(adv_label):
                rooms_res[room_name]['succ'].append(str(attack_dir))

    for room_name in sorted(rooms_res.keys()):
        print(room_name, len(rooms_res[room_name]['succ']))
