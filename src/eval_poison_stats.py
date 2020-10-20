import argparse
import json
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import recognizer.tools as tools
from dataset import Dataset


def get_samples(x, offsets, feature_parameters, context=0):

    """
    Frame and Hops Cheat Sheet
    -----------------------------------
    Samples  |  1  2  3  4  5  6  7  8| 
    -----------------------------------
    Frame 1  |############            |
    Frame 2  |      ############      |
    Frame 3  |            ############|
    -----------------------------------
    Hop 1    |######                  |
    Hop 2    |      ######            |
    Hop 3    |            ######      |
    Hop 4    |                  ######|
    -----------------------------------
    """

    hop_size = feature_parameters['hop_size_samples']
    get_hop = lambda x, offset: x[0, offset * hop_size : (offset+1) * hop_size ]

    # add right context
    right_context = [ offset+c for offset in offsets 
                                for c in range(1, context+1) ]
    # add left context
    left_context = [ offset-c for offset in offsets 
                                for c in range(1, context+1)  ]
    # combine context
    offsets = [ offset for offset in chain(left_context, offsets, right_context) ]

    # to avoid adding the same samples, we add hops rather than frames
    # => add missing hops
    offsets =  [ offset+c for offset in offsets for c in range(2) ]

    # remove double hops and those out of range
    offsets = [ offset for offset in offsets
                        if offset >= 0 and offset < (x.size()[1] // hop_size) ]
    offsets = sorted(set(offsets))

    # finally get the actual samples 
    hops = [ get_hop(x, offset) for offset in offsets ]

    return torch.stack(hops)


def main(exp_dir, dataset):

    targets = exp_dir.rglob('poisons.json')
    stats = defaultdict(dict)

    print("\n[+] collect stats")
    print(f"    @ {exp_dir}")
    for target in tqdm(list(targets),  bar_format='    {l_bar}{bar:30}{r_bar}'):

        if 'TEST-MAN-HJ-7A' not in str(target):
            continue
        try:
            eval_res = list(target.parent.rglob('victim_performance.json'))
            assert len(eval_res) == 1
            eval_res = eval_res[0]
            with open(eval_res) as f:
                eval_res = json.load(f)
            model_pred = eval_res['model_pred']
            adv_digit = target.parent.parent.name.split("-")[-1]
            if model_pred != adv_digit:
                print(f"ignoring f{target}, because the attack has been failed")
                continue
            target_name = target.parents[2].name

            # get itr dirs
            poison_wavs_dir_per_itr = {
                int(itr_dir.name) : itr_dir 
                for itr_dir in target.parent.glob('*')
                if itr_dir.is_dir()
            }
            poison_wavs_dir_per_itr = sorted(poison_wavs_dir_per_itr.items(), key=lambda x: x[0])
            itr, poison_wavs_dir = poison_wavs_dir_per_itr[-1]
            stats[target_name]['max_itr'] = itr

            # get poisons
            poison_info = json.loads(target.read_text())
            poison_wavs = { wav.name.split('.')[0] : wav 
                            for wav in poison_wavs_dir.glob('*.wav') }
            poisons = {}
            for filename in sorted(poison_info.keys()):
                frame_info = poison_info[filename]
                poisons[filename] = [ idx for idxes in frame_info.values() 
                                        for idx in idxes ]

            # get stats
            stats[target_name]['adv_noise'] = []
            stats[target_name]['snr_per_poison_frame'] = []
            stats[target_name]['frames_per_poison'] = []
            stats[target_name]['samples_per_poison'] = []
            stats[target_name]['no_poison'] = len(poisons)

            for filename, offsets in poisons.items():
                _, x_clean, _, _ = dataset[filename]
                x_poison = dataset.load_wav(poison_wavs[filename])

                samples_clean = get_samples(x_clean, offsets, dataset.feature_parameters).reshape(-1)
                samples_poison = get_samples(x_poison, offsets, dataset.feature_parameters).reshape(-1)

                stats[target_name]['frames_per_poison'] += [ len(offsets) ]
                stats[target_name]['samples_per_poison'] += [ len(samples_poison) ]


                v = 0.5648193359375
                # stats[target_name]['adv_noise'] += [ torch.abs(  torch.abs(samples_poison / torch.max(torch.abs(x_poison))) 
                #                                         - torch.abs(samples_clean / torch.max(torch.abs(x_clean)))
                #                                         ).mean().item() ]

                stats[target_name]['adv_noise'] += [ torch.abs(  (torch.abs(samples_poison / v)) 
                                                        - torch.abs(samples_clean / v)
                                                        ).max().item() ]

                stats[target_name]['snr_per_poison_frame'] += [ tools.snrseg(samples_poison.cpu().detach().numpy(), samples_clean.cpu().detach().numpy(), 
                                                                fs=dataset.feature_parameters['sampling_rate'], 
                                                                tf=dataset.feature_parameters['hop_size']) ]

                print(filename, stats[target_name]['snr_per_poison_frame'][-1])


            # get snr all 
            stats[target_name]['snr_per_poison_all'] = []
            for filename, offsets in poisons.items():
                _, x_clean, _, _ = dataset[filename]
                x_poison = dataset.load_wav(poison_wavs[filename])

                stats[target_name]['snr_per_poison_all'] += [ tools.snrseg(x_clean.cpu().detach().numpy(), x_poison.cpu().detach().numpy(), 
                                                fs=dataset.feature_parameters['sampling_rate'], 
                                                tf=dataset.feature_parameters['hop_size']) ]

        except Exception as e:
            print(f"[!] {target_name} @ failed")
            # print(e)

    print("\n[+] General")
    # max itr
    max_itr = [ stat["max_itr"] for stat in stats.values() ]
    print(f'    -> max itr        {np.mean(max_itr):>6.2f} (+-{np.std(max_itr):>6.2f})')
    # max adv. noise
    max_adv_noise = [ np.max(stat['adv_noise']) for stat in stats.values() ]
    print(f'    -> max adv. noise {np.mean(max_adv_noise):6.3f} (+-{np.std(max_adv_noise):>6.3f})')
    # number of poisons files  
    no_poisons = [ stat['no_poison'] for stat in stats.values() ]
    print(f'    -> poison files   {np.mean(no_poisons):>6.0f} (+-{np.std(no_poisons):>6.0f})')
    # number of poisons frames  
    no_of_poison_frames_per_attack = np.array([ np.array(stat['frames_per_poison']).sum() for stat in stats.values() ])
    no_of_poison_samples_per_attack = np.array([ np.array(stat['samples_per_poison']).sum() for stat in stats.values() ])
    print(f'    -> poison frames  {np.mean(no_of_poison_frames_per_attack):6.0f} (+-{np.std(no_of_poison_frames_per_attack):>6.0f})')
    print(f"                     {np.mean(no_of_poison_samples_per_attack  / dataset.feature_parameters['sampling_rate']):>6.2f}s (+-{np.std(no_of_poison_samples_per_attack / dataset.feature_parameters['sampling_rate']):>6.2f})")

    print(f'\n[+] SNRseg')
    snr_per_poison_all = [ np.mean(stat['snr_per_poison_all']) for stat in stats.values() ]
    snr_per_poison_frame = [ np.mean(stat['snr_per_poison_frame']) for stat in stats.values() ]
    print(f"    -> full poison         {np.mean(snr_per_poison_all):>+6.2f} (+-{np.std(snr_per_poison_all):>6.2f})")
    print(f"    -> only poison frames  {np.mean(snr_per_poison_frame):>+6.2f} (+-{np.std(snr_per_poison_frame):>6.2f})")


    for f, stat in stats.items():
        print(f, np.max(stat['adv_noise']), np.mean(stat['snr_per_poison_frame']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--exp-dir', default='/asr-python/tmp/budget-0.03', type=Path)
    parser.add_argument('--model-type', default='', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer'])
    params = parser.parse_args()

    feature_parameters = {'window_size': 25e-3,
                            'hop_size': 12.5e-3,
                            'feature_type': 'raw',
                            'num_ceps': 13,
                            'left_context': 4,
                            'right_context': 4,
                            'sampling_rate': 16000}
    feature_parameters['hop_size_samples'] = tools.sec_to_samples(feature_parameters['hop_size'],
                                                                    feature_parameters['sampling_rate'])
    feature_parameters['window_size_samples'] = tools.next_pow2_samples(feature_parameters['window_size'],
                                                                        feature_parameters['sampling_rate'])

    print('[+] load dataset')
    dataset = Dataset(Path(params.data_dir, params.model_type, 'aligned').joinpath('TRAIN'), feature_parameters)

    main(params.exp_dir, dataset)
