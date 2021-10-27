import os

import argparse as argparse
import torch
import pickle
import argparse
import torchaudio
from pathlib import Path

import recognizer.tools as tools
import rir_simulator.olafilt as olafilt
import rir_simulator.roomsimove_single as roomsimove_single

# # We verified these attacks are just stopped earlier, they didn't crash. For the saek of time, we did not resume
# them until the max step.
stopped_runs = ['_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.02/TEST-MAN-HR-9A/adv-label-1',
                '_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.02/TEST-MAN-IA-9A/adv-label-2',
                '_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.05/TEST-MAN-JH-7A/adv-label-8',
                '_adversarial_paper/perturbation-cap-exp/TwoLayerLight/budget-0.04/eps-0.080/TEST-MAN-IP-3A/adv-label-6'
                ]


def verify_attack(attack_dir):
    attack_dir = str(attack_dir)
    for s in stopped_runs:
        if s in attack_dir:
            return True
    return False


def pred(model, x):
    # predicted transcription
    posteriors = model.features_to_posteriors(x)
    pred_phoneme_seq, _ = model.hmm.posteriors_to_words(posteriors)
    pred_word = tools.str_to_digits(pred_phoneme_seq)
    return pred_word


def calc_rirs():
    for sim_name in SIMS_CFG:
        sim_cfg = SIMS_CFG[sim_name]

        mic_positions = [sim_cfg['mic_pos1'], sim_cfg['mic_pos2']]
        rir = roomsimove_single.do_everything(sim_cfg['room_dim'], mic_positions, sim_cfg['source_pos'], sim_cfg['rt60'])

        SIMS_CFG[sim_name]['rir'] = rir


def overtheairs(x):
    sims = {}
    for sim_name, sim_cfg in SIMS_CFG.items():
        data_rev = olafilt.olafilt(sim_cfg['rir'][:, 0], x.squeeze().detach().cpu().numpy())
        sims[sim_name] = {'cfg': sim_cfg, 'audio': data_rev.T}

    return sims
'''
    'sim0-1': {
        'rt60': 0.4,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.2, 1.7, 2.1],
        'mic_pos2': [2.2, 1.7, 2.1],
        'source_pos': [0.8, 1.2, 1.4],
        'sampling_rate': 16000
    },
    'sim0-2': {
        'rt60': 0.8,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.2, 1.7, 2.1],
        'mic_pos2': [2.2, 1.7, 2.1],
        'source_pos': [0.8, 1.2, 1.4],
        'sampling_rate': 16000
    },
    'sim0-3': {
        'rt60': 1,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.2, 1.7, 2.1],
        'mic_pos2': [2.2, 1.7, 2.1],
        'source_pos': [0.8, 1.2, 1.4],
        'sampling_rate': 16000
    },
'''

'''
CONFIGURATION USED FOR ICML 2021 -- REBUTTAL
SIMS_CFG = {
    'sim1-1': {
        'rt60': 0.4,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.6, 2.3, 2.6],
        'mic_pos2': [2.6, 2.3, 2.6],
        'source_pos': [3.1, 0.7, 1.7],
        'sampling_rate': 16000
    },
    'sim1-2': {
        'rt60': 0.6,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.6, 2.3, 2.6],
        'mic_pos2': [2.6, 2.3, 2.6],
        'source_pos': [3.1, 0.7, 1.7],
        'sampling_rate': 16000
    },
    'sim1-3': {
        'rt60': 0.8,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.6, 2.3, 2.6],
        'mic_pos2': [2.6, 2.3, 2.6],
        'source_pos': [3.1, 0.7, 1.7],
        'sampling_rate': 16000
    },
    'sim1-4': {
        'rt60': 1,
        'room_dim': [4.2, 3.4, 5.2],
        'mic_pos1': [2.6, 2.3, 2.6],
        'mic_pos2': [2.6, 2.3, 2.6],
        'source_pos': [3.1, 0.7, 1.7],
        'sampling_rate': 16000
    },
    'sim2-1': {
        'rt60': 0.4,
        'room_dim': [7.5, 6.3, 6.2],
        'mic_pos1': [5.2, 3.2, 2.5],
        'mic_pos2': [5.2, 3.2, 2.5],
        'source_pos': [7.3, 5.0, 4],
        'sampling_rate': 16000
    },
    'sim2-2': {
        'rt60': 0.6,
        'room_dim': [7.5, 6.3, 6.2],
        'mic_pos1': [5.2, 3.2, 2.5],
        'mic_pos2': [5.2, 3.2, 2.5],
        'source_pos': [7.3, 5.0, 4],
        'sampling_rate': 16000
    },
    'sim2-3': {
        'rt60': 0.8,
        'room_dim': [7.5, 6.3, 6.2],
        'mic_pos1': [5.2, 3.2, 2.5],
        'mic_pos2': [5.2, 3.2, 2.5],
        'source_pos': [7.3, 5.0, 4],
        'sampling_rate': 16000
    },
    'sim2-4': {
        'rt60': 1,
        'room_dim': [7.5, 6.3, 6.2],
        'mic_pos1': [5.2, 3.2, 2.5],
        'mic_pos2': [5.2, 3.2, 2.5],
        'source_pos': [7.3, 5.0, 4],
        'sampling_rate': 16000
    },

}
'''

SIMS_CFG = {
    'sim1-1': {
        'rt60': 0.4,
        'room_dim': [10.7, 6.9, 2.6],
        'mic_pos1': [1, 4.5, 1.3],
        'mic_pos2': [1, 4.5, 1.3],
        'source_pos': [8.1, 3.3, 1.4],
        'sampling_rate': 16000
    },
    'sim1-2': {
        'rt60': 0.6,
        'room_dim': [10.7, 6.9, 2.6],
        'mic_pos1': [1, 4.5, 1.3],
        'mic_pos2': [1, 4.5, 1.3],
        'source_pos': [8.1, 3.3, 1.4],
        'sampling_rate': 16000
    },
    'sim1-3': {
        'rt60': 0.8,
        'room_dim': [10.7, 6.9, 2.6],
        'mic_pos1': [1, 4.5, 1.3],
        'mic_pos2': [1, 4.5, 1.3],
        'source_pos': [8.1, 3.3, 1.4],
        'sampling_rate': 16000
    },
    'sim1-4': {
        'rt60': 1,
        'room_dim': [10.7, 6.9, 2.6],
        'mic_pos1': [1, 4.5, 1.3],
        'mic_pos2': [1, 4.5, 1.3],
        'source_pos': [8.1, 3.3, 1.4],
        'sampling_rate': 16000
    },
    'sim2-1': {
        'rt60': 0.4,
        'room_dim': [4.6, 6.9, 3.1],
        'mic_pos1': [3.8, 3.2, 1.2],
        'mic_pos2': [3.8, 3.2, 1.2],
        'source_pos': [3.8, 5.3, 1],
        'sampling_rate': 16000
    },
    'sim2-2': {
        'rt60': 0.6,
        'room_dim': [4.6, 6.9, 3.1],
        'mic_pos1': [3.8, 3.2, 1.2],
        'mic_pos2': [3.8, 3.2, 1.2],
        'source_pos': [3.8, 5.3, 1],
        'sampling_rate': 16000
    },
    'sim2-3': {
        'rt60': 0.8,
        'room_dim': [4.6, 6.9, 3.1],
        'mic_pos1': [3.8, 3.2, 1.2],
        'mic_pos2': [3.8, 3.2, 1.2],
        'source_pos': [3.8, 5.3, 1],
        'sampling_rate': 16000
    },
    'sim2-4': {
        'rt60': 1,
        'room_dim': [4.6, 6.9, 3.1],
        'mic_pos1': [3.8, 3.2, 1.2],
        'mic_pos2': [3.8, 3.2, 1.2],
        'source_pos': [3.8, 5.3, 1],
        'sampling_rate': 16000
    },
    'sim3-1': {
        'rt60': 0.4,
        'room_dim': [7.5, 4.6, 3.1],
        'mic_pos1': [0.4, 0.9, 1.1],
        'mic_pos2': [0.4, 0.9, 1.1],
        'source_pos': [6.9, 1.9, 2.6],
        'sampling_rate': 16000
    },
    'sim3-2': {
        'rt60': 0.6,
        'room_dim': [7.5, 4.6, 3.1],
        'mic_pos1': [0.4, 0.9, 1.1],
        'mic_pos2': [0.4, 0.9, 1.1],
        'source_pos': [6.9, 1.9, 2.6],
        'sampling_rate': 16000
    },
    'sim3-3': {
        'rt60': 0.8,
        'room_dim': [7.5, 4.6, 3.1],
        'mic_pos1': [0.4, 0.9, 1.1],
        'mic_pos2': [0.4, 0.9, 1.1],
        'source_pos': [6.9, 1.9, 2.6],
        'sampling_rate': 16000
    },
    'sim3-4': {
        'rt60': 1,
        'room_dim': [7.5, 4.6, 3.1],
        'mic_pos1': [0.4, 0.9, 1.1],
        'mic_pos2': [0.4, 0.9, 1.1],
        'source_pos': [6.9, 1.9, 2.6],
        'sampling_rate': 16000
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--attack-root-dir',
                        default='_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.04', type=Path)

    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    # parser.add_argument('--seed', default=21212121, type=int)
    parser.add_argument('--victim-net', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer'])

    params = parser.parse_args()

    attack_dirs = list(params.attack_root_dir.iterdir())
    attack_dirs_tmp = [list(list(a.iterdir())[0].iterdir())[0] for a in attack_dirs]
    for idx, a in enumerate(attack_dirs_tmp):
        steps = [int(x.name) for x in a.iterdir() if x.is_dir()]
        max_step = max(steps)
        attack_dirs[idx] = a.joinpath(f"{max_step}")

    sims_saved_path = Path('asr-python/src/rir_simulator/sims_cfg.h5')
    if sims_saved_path.exists():
        SIMS_CFG = pickle.load(sims_saved_path.open('rb'))
        print("rirs loaded")
    else:
        calc_rirs()
        pickle.dump(SIMS_CFG, sims_saved_path.open('wb'))
        print("rirs saved")
        
    params.data_dir = params.data_dir.joinpath(params.victim_net)

    sims_res = {sim_name: {'succ': []} for sim_name in SIMS_CFG}
    sims_res['original'] = {'succ': []}
    print("In total, we have {} attack examples!".format(len(list(attack_dirs))))
    for attack_dir in attack_dirs:
        print("----")
        if params.victim_net in str(attack_dir):
            eval_res_path = attack_dir.joinpath("new-hmm")
        else:
            eval_res_path = attack_dir.joinpath(f"{params.victim_net}-new-hmm")
        assert os.path.exists(attack_dir)

        # attack_step_dirs = [s for s in attack_dir.iterdir() if s.is_dir()]
        # attack_step_dirs = sorted(attack_step_dirs, key=lambda s: int(s.name))
        # attack_last_step_dir = attack_step_dirs[-1]

        # last_step_num = int(attack_dir.name)
        # if last_step_num != 20:
        #     with open(attack_dir.joinpath('log.txt')) as f:
        #         log = f.read()
        #         assert 'Evaluating the victim - 3' in log or verify_attack(attack_dir)
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

        if len(pred_word) == 1 and str(pred_word[0]) == str(adv_label):
            sims_res['original']['succ'].append(str(attack_dir))

        print("Now let's see what happens for the over-the-air simulations")
        target_sims = overtheairs(target_x)

        for sim_name, sim in target_sims.items():
            x = torch.tensor(sim['audio']).cuda().float().unsqueeze(dim=0)
            pred_word = pred(model, x)
            print("Simulated targeted audio file is identified as: {}".format(pred_word))

            if len(pred_word) == 1 and str(pred_word[0]) == str(adv_label):
                sims_res[sim_name]['succ'].append(str(attack_dir))

            target_sim_path = eval_res_path / f"target-{sim_name}.wav"
            torchaudio.save(str(target_sim_path), x.detach().cpu(), 16000)

    for sim_name in sorted(sims_res.keys()):
        print(sim_name, len(sims_res[sim_name]['succ']))
