import os
import sys
import json
import time
import torch
import pickle
import shutil
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

from pathlib import Path
from target import Target

import recognizer.tools as tools
from recognizer.model import init_model
import recognizer.hmm as HMM
from dataset import Dataset

# # We verified these attacks are just stopped earlier, they didn't crash. For the saek of time, we did not resume
# them until the max step.
stopped_runs = ['_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.02/TEST-MAN-HR-9A/adv-label-1',
                '_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.02/TEST-MAN-IA-9A/adv-label-2',
                '_adversarial_paper/one-digit-exp/TwoLayerPlus/budget-0.05/TEST-MAN-JH-7A/adv-label-8',
                '_adversarial_paper/perturbation-cap-exp/TwoLayerLight/budget-0.04/eps-0.080/TEST-MAN-IP-3A/adv-label-6'
                ]

VICTIM_CONFIGS = {
    'cfg1': {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'epochs': '10N-1V-20N'
    },
    'cfg2': {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'epochs': '10N-1V-20N'
    },
    'cfg2-dp-0.2': {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'dropout': 0.2,
        'epochs': '10N-1V-20N',
    }
}


def verify_attack(attack_dir):
    attack_dir = str(attack_dir)
    for s in stopped_runs:
        if s in attack_dir:
            return True
    return False


class Poisons:

    def __init__(self, data_dir, poison_paths, poison_info_path):
        super(Poisons, self).__init__()

        self.filenames = [p.stem.split(".poison")[0] for p in poison_paths]

        self._poisons_X = {}
        self._poisons_Y = {}
        for p in poison_paths:
            filename = p.stem.split(".poison")[0]
            x, _ = torchaudio.load(p)
            x = x.cuda()
            y = torch.load(data_dir.joinpath("Y", filename).with_suffix(".pt")).cuda()

            self._poisons_X[filename] = x
            self._poisons_Y[filename] = y

        with open(poison_info_path) as f:
            tmp = json.load(f)

        self._poisons_frames = defaultdict(dict)
        self._target_idx_to_poison_files = defaultdict(list)
        for filename, frames in tmp.items():
            for target_idx, frame_indices in frames.items():
                target_idx = int(target_idx)
                self._poisons_frames[filename][target_idx] = torch.tensor(frame_indices).cuda()
                self._target_idx_to_poison_files[target_idx] += [filename]

    def update_poison_y_label(self, filename, y):
        self._poisons_Y[filename] = y

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

    def get_poisons_frames(self):

        for target_idx, poison_files in self._target_idx_to_poison_files.items():
            poison_files = sorted(poison_files)
            poisons_x = [self._poisons_X[p] for p in poison_files]

            max_x_l = max([x.shape[1] for x in poisons_x])
            poisons_x = [F.pad(x, pad=(0, max_x_l - x.shape[1], 0, 0)) for x in poisons_x]
            poisons_x = torch.cat(poisons_x)

            poisons_frames_indices = [self._poisons_frames[poison_file][target_idx] for poison_file in poison_files]

            yield target_idx, poisons_x, poisons_frames_indices

    def is_poison(self, filename):
        return filename in self._poisons_X

    def get_poison(self, filename):
        return self._poisons_X[filename]

    @property
    def X(self):
        return [x for x in self._poisons_X.values()]


def copy_training_dataset_raw(src, dst, poison_paths):
    dst.mkdir(parents=True, exist_ok=True)
    poison_filenames = [p.stem.split(".poison")[0] for p in poison_paths]

    # First copy the poison samples
    poison_filenames = []
    for p in poison_paths:
        filename = p.stem.split(".poison")[0]
        poison_filenames.append(filename)
        shutil.copyfile(p, dst.joinpath(filename).with_suffix(".wav"))

    # Then copy other clean samples, no need to copy them!
    for sample_path in src.glob("*.wav"):
        sample_filename = sample_path.stem
        if sample_filename not in poison_filenames:
            shutil.copyfile(sample_path, dst.joinpath(sample_path.name))

    # # Then create symlinks for other clean samples, no need to copy them!
    # for sample_path in src.glob("*.wav"):
    #     sample_filename = sample_path.stem
    #     if sample_filename not in poison_filenames:
    #         os.symlink(sample_path, dst.joinpath(sample_path.name))


def copy_test_dataset_raw(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for sample_path in src.glob("*.wav"):
        os.symlink(sample_path, dst.joinpath(sample_path.name))


# def eval_victim(model_type, feature_parameters, dataset, dataset_test, target, dropout):
#     res = {}
#
#     model = init_model(model_type, feature_parameters, dataset.hmm, dropout=dropout).cuda()
#     model.train_model(dataset, epochs=15)
#
#     # benign accuracy of victim model
#     model_acc, test_res = model.parallel_test(dataset_test)
#
#     # Poisons Classification Loss.
#     poisons_x, poisons_y, poisons_true_length, poisons_imp_indices, _ = dataset.poisons.get_all_poisons()
#     poisons_classification_loss, poisons_imp_indices_classification_loss = \
#         model.compute_loss_batch(poisons_x, poisons_y, poisons_true_length, important_indices=poisons_imp_indices)
#
#     # similarity to original and target states
#     loss_original_states = model.compute_loss_single(target.x, target.original_states).item()
#     loss_adversarial_states = model.compute_loss_single(target.x, target.adversarial_states, target.adv_indices).item()
#
#     # predicted transcription
#     posteriors = model.features_to_posteriors(target.x)
#     pred_phoneme_seq, victim_states_viterbi = model.hmm.posteriors_to_words(posteriors)
#     pred_phoneme_seq = tools.str_to_digits(pred_phoneme_seq)
#     victim_states_argmax = np.argmax(posteriors, axis=1)
#
#     # target states
#     target_states = target.adversarial_states
#
#     victim_adv_states_acc = (100.0 *
#                              sum([v == t for v, t in zip(victim_states_argmax[target.adv_indices],
#                                                          target_states[target.adv_indices].tolist())])) / len(
#         target.adv_indices)
#
#     # Bullseye Loss (just for evaluation!)
#     from craft_poisons import bullseye_loss
#     victim_bullseye_loss = bullseye_loss(target, dataset.poisons, [model], compute_gradients=False)
#
#     # logging
#     print(f'')
#     print(f'    -> loss original states              : {loss_original_states:6.4f}')
#     print(f'    -> loss adversarial states           : {loss_adversarial_states:6.4f}')
#     print(f'    -> clean accuracy                    : {model_acc:6.4f}')
#     print(f'    -> poisons cls. loss                 : {poisons_classification_loss.item():6.4f}')
#     print(f'    -> imp. indices poisons cls. loss    : {poisons_imp_indices_classification_loss.item():6.4f}')
#     print(f'    -> bullseye loss                     : {victim_bullseye_loss:6.4f}')
#     print(f'    -> adv. states acc.                  : {victim_adv_states_acc:6.4f}')
#     print(f"    -> model decoded                     : {', '.join([f'{p:>3}' for p in pred_phoneme_seq])}")
#     print(f"    -> target label                      : {', '.join([f'{p:>3}' for p in target.target_transcription])}")
#     print(f"    -> model output\n")
#     states_to_interval = lambda states: [states[i:i + 28] for i in range(0, len(states), 28)]
#     for original_seq, target_seq, victim_argmax_seq, victim_viterbi_seq in \
#             zip(states_to_interval(target.original_states.tolist()), states_to_interval(target_states.tolist()),
#                 states_to_interval(victim_states_argmax), states_to_interval(victim_states_viterbi)):
#         print("       " + "| ORIGINAL  " + " ".join([f'{x:2}' for x in original_seq]))
#         print("       " + "| TARGET    " + " ".join([f'{x:2}' for x in target_seq]))
#         print("       " + "| ARGMAX    " + " ".join([f'{x:2}' for x in victim_argmax_seq]))
#         print("       " + "| VITERBI   " + " ".join([f'{x:2}' for x in victim_viterbi_seq]))
#         print('')
#
#     res = {"loss_original_states": loss_original_states,
#            "loss_adversarial_states": loss_adversarial_states,
#            # "model_clean_test_acc": model_acc,
#            "poisons_classification_loss": poisons_classification_loss.item(),
#            "poisons_imp_indices_classification_loss": poisons_imp_indices_classification_loss.item(),
#            "bullseye_loss": victim_bullseye_loss,
#            "adv_states_acc": victim_adv_states_acc,
#            "model_pred": "".join([str(p) for p in pred_phoneme_seq]),
#            "test_res": test_res
#            }
#
#     return model, res


def eval_victim_viterbi_scratch(args, dataset, dataset_test, target_x, target_filename, adv_label, model):
    res = {}

    # benign accuracy of victim model
    model_acc, test_res = model.parallel_test(dataset_test)

    # Poisons Classification Loss.
    poisons_x, poisons_y, poisons_true_length, poisons_imp_indices, _ = dataset.poisons.get_all_poisons()
    poisons_classification_loss, poisons_imp_indices_classification_loss = \
        model.compute_loss_batch(poisons_x, poisons_y, poisons_true_length, important_indices=poisons_imp_indices)

    target_x = target_x.cuda()
    # predicted transcription
    posteriors = model.features_to_posteriors(target_x)
    pred_phoneme_seq, victim_states_viterbi = model.hmm.posteriors_to_words(posteriors)
    if pred_phoneme_seq == -1:
        pred_phoneme_seq = ['']
    pred_phoneme_seq = [p.lower() for p in pred_phoneme_seq]
    # pred_phoneme_seq = tools.str_to_digits(pred_phoneme_seq)
    victim_states_argmax = np.argmax(posteriors, axis=1)

    if args.task == 'TIDIGITS':
        speaker = "-".join(target_filename.split("-")[:-1])
        files, xs, ys, texts = dataset_test.get_speaker_utterances(speaker)
        speaker_res = {}
        for f, x, y, l in zip(files, xs, ys, texts):
            x = x.cuda()
            y = y.cuda()
            post = model.features_to_posteriors(x)
            pred_word_seq, victim_states_viterbi = model.hmm.posteriors_to_words(post)
            speaker_res[f] = {'pred_word_seq': ' '.join(pred_word_seq), 'best_path': victim_states_viterbi.tolist(),
                              'label_word_seq': l}
    elif args.task == 'SPEECHCOMMANDS':
        speaker_res = {}
    else:
        assert False

    # logging
    print(f'')
    print(f'    -> clean accuracy                    : {model_acc:6.4f}')
    print(f'    -> poisons cls. loss                 : {poisons_classification_loss.item():6.4f}')
    print(f'    -> imp. indices poisons cls. loss    : {poisons_imp_indices_classification_loss.item():6.4f}')
    print(f"    -> model decoded                     : {pred_phoneme_seq}")
    print(f"    -> target label                      : {adv_label}")
    print(f"    -> model output\n")

    res = {
        # "model_clean_test_acc": model_acc,
        "poisons_classification_loss": poisons_classification_loss.item(),
        "poisons_imp_indices_classification_loss": poisons_imp_indices_classification_loss.item(),
        "model_pred": "".join([str(p) for p in pred_phoneme_seq]),
        "test_res": test_res,
        "speaker_res": speaker_res
    }

    return res


def preprocess(task, raw_clean_data_dir, poisoned_dataset_path, feature_parameters, poison_samples_paths, speakers_list):
    def load_raw_data_dir(dataset_dir, poison_samples_paths, speakers_list=None, device='cpu'):
        dataset_dir = dataset_dir.resolve()  # To resolve symlinks!
        # find raw data
        wav_files = [f for f in sorted(dataset_dir.joinpath('wav').resolve().glob('*.wav'))]
        praat_files = [f for f in sorted(dataset_dir.joinpath('TextGrid').resolve().glob('*.TextGrid'))]
        lab_files = [f for f in sorted(dataset_dir.joinpath('lab').resolve().glob('*.lab'))]

        # load raw data
        X = []
        Y = []
        texts = []
        wav_files_selected = []
        for wav_file, praat_file, lab_file in tqdm(zip(wav_files, praat_files, lab_files),
                                                   total=len(wav_files),
                                                   bar_format='    load raw     {l_bar}{bar:30}{r_bar}'):
            # sanity check
            assert wav_file.stem == praat_file.stem == lab_file.stem, f'{wav_file.stem} {praat_file.stem} {lab_file.stem}'

            if speakers_list and '-'.join(wav_file.stem.split("-")[-3:-1]) not in speakers_list and wav_file.stem not in poison_samples_paths:
                # This is not the poisoned data. It is also not spoken by the speakers list (chosen for the victim). So
                # it should not be in the victim's training set
                continue

            wav_files_selected.append(wav_file)
            if wav_file.stem in poison_samples_paths:
                # This is a poison sample, we should not load the original version then!
                wav_file = poison_samples_paths[wav_file.stem]
            ## load x
            x, _ = torchaudio.load(wav_file)
            # round to the next `full` frame
            num_frames = np.floor(x.shape[1] / hop_size_samples)
            x = x[:, :int(num_frames * hop_size_samples)].to(device)
            X.append(x)
            ## load y
            # optional: convert praats into jsons
            # dataset_dir.joinpath('align').mkdir(parents=True, exist_ok=True)
            # tg = tgio.openTextgrid(praat_file)
            # align_dict = tools.textgrid_to_dict(tg)
            # json_file = Path(str(praat_file).replace('TextGrid', 'align')).with_suffix('.json')
            # json_file.write_text(json.dumps(align_dict, indent=4))
            # y = tools.json_file_to_target(json_file, sampling_rate, window_size_samples, hop_size_samples, hmm)
            y = tools.praat_file_to_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm)
            y = torch.from_numpy(y).to(device)
            Y.append(y)
            ## load text
            text = lab_file.read_text().strip()
            texts.append(text)

        return wav_files_selected, X, Y, texts

    assert raw_clean_data_dir.is_dir()

    poison_samples_paths = {str(p.stem).split(".poison")[0]: p for p in poison_samples_paths}

    # data config
    sampling_rate = feature_parameters['sampling_rate']
    window_size_samples = tools.next_pow2_samples(feature_parameters['window_size'], sampling_rate)
    hop_size_samples = tools.sec_to_samples(feature_parameters['hop_size'], sampling_rate)

    poisoned_plain_out_dir = Path(poisoned_dataset_path).joinpath('plain')
    poisoned_plain_out_dir.mkdir()

    hmm = HMM.HMM(task, 'word')
    pickle.dump(hmm, poisoned_plain_out_dir.joinpath('hmm.h5').open('wb'))

    # pre-proccess plain data
    dataset_names = [d.name for d in Path(raw_data_dir).glob('*') if d.is_dir()]
    for dataset_name in dataset_names:
        if 'train' in dataset_name.lower():
            if speakers_list is not None:
                speakers_list = set(speakers_list)
            wav_files, X, Y, texts = load_raw_data_dir(raw_data_dir.joinpath(dataset_name), poison_samples_paths, speakers_list)
        elif dataset_name.lower() == 'test':
            wav_files, X, Y, texts = load_raw_data_dir(raw_data_dir.joinpath(dataset_name), poison_samples_paths)
        else:
            continue
        ## dump plain
        X_out_dir = poisoned_plain_out_dir.joinpath(dataset_name, 'X');
        X_out_dir.mkdir(parents=True)
        Y_out_dir = poisoned_plain_out_dir.joinpath(dataset_name, 'Y');
        Y_out_dir.mkdir(parents=True)
        text_out_dir = poisoned_plain_out_dir.joinpath(dataset_name, 'text');
        text_out_dir.mkdir(parents=True)
        wav_out_dir = poisoned_plain_out_dir.joinpath(dataset_name, 'wavs');
        wav_out_dir.mkdir(parents=True)
        for wav_file, x, y, text in tqdm(zip(wav_files, X, Y, texts),
                                         total=len(wav_files), bar_format='    dump plain  {l_bar}{bar:30}{r_bar}'):
            filename = wav_file.stem
            torch.save(y, Y_out_dir.joinpath(filename).with_suffix('.pt'))
            torch.save(x, X_out_dir.joinpath(filename).with_suffix('.pt'))
            text_out_dir.joinpath(filename).with_suffix('.txt').write_text(text)
            shutil.copyfile(wav_file, wav_out_dir.joinpath(filename).with_suffix('.wav'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--attack-dir',
                        default='_adversarial_paper2/one-digit-exp/eps--1.00/TEST-MAN-IP-5A/adv-label-6')

    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=21212121, type=int)
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="enables the dropout of the models, in training and also afterwards")
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--viterbi-scratch', default=True)
    parser.add_argument('--model-type', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer', 'FourLayerPlus'])
    parser.add_argument('--epochs', default='15N-3V-15N', choices=['15N-3V-15N', '10N-2V-20N'])

    parser.add_argument('--victim-config', default=None, choices=['cfg1', 'cfg2', 'cfg2-dp-0.2'])

    parser.add_argument('--speakers-list-path', default='/asr-python/speakers-training.txt')
    parser.add_argument('--speaker-start-index', default=-1, type=int)
    parser.add_argument('--speaker-end-index', default=-1, type=int)
    parser.add_argument('--task', default='TIDIGITS', choices=['TIDIGITS', 'SPEECHCOMMANDS'])

    params = parser.parse_args()

    if params.victim_config is not None:
        print("WARNING!!!!")
        print(f"{params.victim_config} is being used for victim evaluation!")
        configs = VICTIM_CONFIGS[params.victim_config]
        for param, value in configs.items():
            prev_value = getattr(params, param)
            setattr(params, param, value)
            print(f"Parameter {param} is changed from {prev_value} to {value}")

    if params.task == 'TIDIGITS':
        assert os.path.exists(params.speakers_list_path), params.speakers_list_path
        with open(params.speakers_list_path) as f:
            speakers_list = f.readlines()
            speakers_list = [s.strip() for s in speakers_list]
        if params.speaker_start_index != -1 or params.speaker_end_index != -1:
            assert 0 <= params.speaker_start_index and params.speaker_end_index <= len(speakers_list)
            speakers_list = speakers_list[params.speaker_start_index:params.speaker_end_index + 1]
            speakers_split_identifier = 'speakers-{}-{}'.format(params.speaker_start_index, params.speaker_end_index)
        else:
            speakers_split_identifier = 'speakers-all'
    elif params.task == 'SPEECHCOMMANDS':
        speakers_split_identifier = 'speakers-none'
        speakers_list = None
    else:
        assert False

    params.data_dir = params.data_dir.joinpath(params.model_type)

    # assert params.model_type in params.attack_dir, "It seems you are trying to evalute " \
    #                                                "results generated for a different model type"
    # assert params.model_type in str(params.data_dir), "You are using the wrong hmm (and aligned data)!"

    feature_parameters = {'window_size': 25e-3,
                          'hop_size': 12.5e-3,
                          'feature_type': 'raw',
                          'num_ceps': 13,
                          'left_context': 4,
                          'right_context': 4,
                          'sampling_rate': tools.get_sampling_rate(params.data_dir.parent)}
    feature_parameters['hop_size_samples'] = tools.sec_to_samples(feature_parameters['hop_size'],
                                                                  feature_parameters['sampling_rate'])
    feature_parameters['window_size_samples'] = tools.next_pow2_samples(feature_parameters['window_size'],
                                                                        feature_parameters['sampling_rate'])

    tools.set_seed(params.seed)

    attack_dir = Path(params.attack_dir)

    assert os.path.exists(attack_dir), print(f'attack dir does not exits {attack_dir}')

    if not attack_dir.joinpath('log.txt').is_file():
        assert len(list(attack_dir.iterdir())) == 1, "more than one instance of attack exist!"
        attack_dir = list(attack_dir.iterdir())[0]

    attack_step_dirs = [s for s in attack_dir.iterdir() if s.is_dir()]
    attack_step_dirs = sorted(attack_step_dirs, key=lambda s: int(s.name))
    attack_last_step_dir = attack_step_dirs[-1]

    last_step_num = int(attack_last_step_dir.name)
    if last_step_num != 20:
        with open(attack_dir.joinpath('log.txt')) as f:
            log = f.read()
            if 'Evaluating the victim - 3' not in log:
                print(f"early stopping happened at step: {last_step_num}")
    adv_label = attack_last_step_dir.parent.parent.name.split("adv-label-")[1]
    target_filename = attack_last_step_dir.parent.parent.parent.name

    poison_paths = sorted(attack_last_step_dir.glob("*.wav"))

    if params.viterbi_scratch:
        if params.victim_config is not None:
            eval_res_path = attack_last_step_dir / f"victim-{params.victim_config}"
        else:
            eval_res_path = attack_last_step_dir

        if params.model_type in params.attack_dir:
            # The victim's architecture is the same as the surrogate networks.
            if speakers_split_identifier == 'speakers-all':
                eval_res_path = eval_res_path / "new-hmm"
            else:
                eval_res_path = eval_res_path / f"{speakers_split_identifier}-new-hmm"
        else:
            print("The victim uses a different network architecture: {}".format(params.model_type))
            if speakers_split_identifier == 'speakers-all':
                eval_res_path = eval_res_path / f"{params.model_type}-new-hmm"
            else:
                eval_res_path = eval_res_path / f"{speakers_split_identifier}-{params.model_type}-new-hmm"

        if eval_res_path.joinpath("victim_performance.json").exists():
            sys.exit()

        if True: 
            eval_res_path.mkdir(parents=True)

            poisoned_dataset_path = eval_res_path / "poisoned_dataset"
            poisoned_dataset_path = poisoned_dataset_path / "data"
            #
            assert not poisoned_dataset_path.exists()
            poisoned_dataset_path.mkdir(parents=True)

            # This copies the poison files, and symlink other clean files, to save space!
            raw_data_dir = params.data_dir.parent.joinpath("raw")

            # This only generates files in the plain folder, aligned based on a new HMM.
            preprocess(params.task, raw_data_dir, poisoned_dataset_path, feature_parameters, poison_paths, speakers_list)

            dataset = Dataset(poisoned_dataset_path.joinpath('plain', 'TRAIN'), feature_parameters, params.seed)
            dataset_test = Dataset(poisoned_dataset_path.joinpath('plain', 'TEST'), feature_parameters, params.seed)
            dataset.poisons = Poisons(poisoned_dataset_path.joinpath('plain', 'TRAIN'), poison_paths,
                                      attack_dir.joinpath("poisons").with_suffix(".json"))
            # Training the model
            model = init_model(params.model_type, feature_parameters, dataset.hmm, dropout=params.dropout)

            if params.epochs == '15N-3V-15N':
                model.train_model(dataset, epochs=15, batch_size=params.batch_size, lr=params.learning_rate)
                model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True,
                                  lr=params.learning_rate)
                model.hmm.A = model.hmm.modifyTransitions(model.hmm.A_count)
                model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True,
                                  lr=params.learning_rate)
                model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True,
                                  lr=params.learning_rate, update_y_label=True)
                model.train_model(dataset, epochs=15, batch_size=params.batch_size, lr=params.learning_rate)
            elif params.epochs == '10N-1V-20N':
                model.train_model(dataset, epochs=10, batch_size=params.batch_size, lr=params.learning_rate)
                # model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True,
                #                   lr=params.learning_rate)
                model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True,
                                  lr=params.learning_rate, update_y_label=True)
                model.hmm.A = model.hmm.modifyTransitions(model.hmm.A_count)
                model.train_model(dataset, epochs=20, batch_size=params.batch_size, lr=params.learning_rate)
            else:
                assert False

            pickle.dump(model.hmm, eval_res_path.joinpath('aligned-hmm.h5').open('wb'))
            pickle.dump(model, eval_res_path.joinpath('model.h5').open('wb'))

        else:
            poisoned_dataset_path = eval_res_path / "poisoned_dataset"
            poisoned_dataset_path = poisoned_dataset_path / "data"
            
            dataset = Dataset(poisoned_dataset_path.joinpath('plain', 'TRAIN'), feature_parameters, params.seed)
            dataset_test = Dataset(poisoned_dataset_path.joinpath('plain', 'TEST'), feature_parameters, params.seed)
            dataset.poisons = Poisons(poisoned_dataset_path.joinpath('plain', 'TRAIN'), poison_paths,
                                      attack_dir.joinpath("poisons").with_suffix(".json"))

            hmm = pickle.load(eval_res_path.joinpath('aligned-hmm.h5').open('rb'))
            model = pickle.load(eval_res_path.joinpath('model.h5').open('rb'))

        target_x_filepath = poisoned_dataset_path.joinpath('plain', 'TEST', 'X', target_filename).with_suffix('.pt')
        # target_y_filepath = params.data_dir.joinpath('plain', 'TEST', 'Y', target_filename).with_suffix('.pt')
        # target_text = params.data_dir.joinpath('plain', 'TEST', 'text', target_filename).with_suffix(
        #     '.txt').read_text()
        assert target_x_filepath.exists()
        # assert target_y_filepath.exists()

        target_x = torch.load(target_x_filepath)

        test_res = eval_victim_viterbi_scratch(params, dataset, dataset_test, target_x, target_filename,
                                               adv_label, model)

        shutil.rmtree(poisoned_dataset_path)
        print("---------ATTENTION--------")
        print(target_filename)
        print(adv_label)
        print(test_res['model_pred'])
        # print(test_res['test_res'][target_filename]['pred_word_seq'])
        print("--------------------------")

        with open(eval_res_path.joinpath("victim_performance.json"), "w") as f:
            json.dump(test_res, f)
