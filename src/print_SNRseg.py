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
    get_hop = lambda x, offset: x[0, offset * hop_size: (offset + 1) * hop_size]

    # add right context
    right_context = [offset + c for offset in offsets
                     for c in range(1, context + 1)]
    # add left context
    left_context = [offset - c for offset in offsets
                    for c in range(1, context + 1)]
    # combine context
    offsets = [offset for offset in chain(left_context, offsets, right_context)]

    # to avoid adding the same samples, we add hops rather than frames
    # => add missing hops
    offsets = [offset + c for offset in offsets for c in range(2)]

    # remove double hops and those out of range
    offsets = [offset for offset in offsets
               if offset >= 0 and offset < (x.size()[1] // hop_size)]
    offsets = sorted(set(offsets))

    # finally get the actual samples 
    hops = [get_hop(x, offset) for offset in offsets]

    return torch.stack(hops)


def main(poison_dir, dataset, model_type):
    v = max([torch.max(x) for x in dataset.X])
    print(v)

    poison_json_path = poison_dir / "poisons.json"
    poison_wavs_dir_per_itr = {
        int(itr_dir.name): itr_dir
        for itr_dir in poison_dir.glob('*')
        if itr_dir.is_dir()
    }
    poison_wavs_dir_per_itr = sorted(poison_wavs_dir_per_itr.items(), key=lambda x: x[0])
    itr, poison_wavs_dir = poison_wavs_dir_per_itr[-1]

    # get poisons
    poison_info = json.loads(poison_json_path.read_text())
    poison_wavs = {wav.name.split('.')[0]: wav
                   for wav in poison_wavs_dir.glob('*.wav')}
    poisons = {}
    for filename in sorted(poison_info.keys()):
        frame_info = poison_info[filename]
        poisons[filename] = [idx for idxes in frame_info.values()
                             for idx in idxes]

    target_name = ['does not matter']

    # get stats
    adv_noise = []
    snr_per_poison_frame = []
    frames_per_poison = []
    samples_per_poison = []
    no_poison = len(poisons)

    for filename, offsets in poisons.items():
        # if filename not in selected_files['poisons'][target_name]:
        #     continue
        _, x_clean, _, _ = dataset[filename]
        x_poison = dataset.load_wav(poison_wavs[filename])

        samples_clean = get_samples(x_clean, offsets, dataset.feature_parameters).reshape(-1).cuda()
        samples_poison = get_samples(x_poison, offsets, dataset.feature_parameters).reshape(-1).cuda()

        frames_per_poison += [len(offsets)]
        samples_per_poison += [len(samples_poison)]

        # stats[target_name]['adv_noise'] += [ torch.abs(  torch.abs(samples_poison / torch.max(torch.abs(x_poison)))
        #                                         - torch.abs(samples_clean / torch.max(torch.abs(x_clean)))
        #                                         ).mean().item() ]

        adv_noise += [torch.abs((torch.abs(samples_poison / v))
                                - torch.abs(samples_clean / v)
                                ).max().item()]

        snr_per_poison_frame += [
            tools.snrseg(samples_poison.cpu().detach().numpy(), samples_clean.cpu().detach().numpy(),
                         fs=dataset.feature_parameters['sampling_rate'],
                         tf=dataset.feature_parameters['hop_size'])]

    # get snr all
    snr_per_poison_all = []
    for filename, offsets in poisons.items():
        _, x_clean, _, _ = dataset[filename]
        x_poison = dataset.load_wav(poison_wavs[filename])

        snr_per_poison_all += [tools.snrseg(x_clean.cpu().detach().numpy(), x_poison.cpu().detach().numpy(),
                                            fs=dataset.feature_parameters['sampling_rate'],
                                            tf=dataset.feature_parameters['hop_size'])]

    print("\n[+] General")
    # max adv. noise
    print(f'    -> max adv. noise {np.max(adv_noise):>+6.4f}')

    print(f'\n[+] SNRseg')
    print(f"    -> full poison         {np.mean(snr_per_poison_all):>+6.2f}")
    print(f"    -> only poison frames  {np.mean(snr_per_poison_frame):>+6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--poison-dir', default='/asr-python/tmp/budget-0.03', type=Path)
    parser.add_argument('--model-type', default='', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer', 'ThreeLayerPlusPlus'])
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
    dataset = Dataset(Path(params.data_dir, params.model_type, 'plain').joinpath('TRAIN'), feature_parameters)

    main(params.poison_dir, dataset, params.model_type)
