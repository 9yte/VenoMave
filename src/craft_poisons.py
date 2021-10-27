import argparse
import logging
import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
from itertools import product

import numpy as np
from tqdm import tqdm
import random
import recognizer.tools as tools
import torch
import torch.multiprocessing as mp
from dataset import Dataset, preprocess_dataset
from poisons import Poisons
from recognizer.model import init_model
from target import Target

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)


class ModelWrapper:
    def __init__(self, model_index, crafting_step, seed, model_type, feature_parameters,
                 dataset, device, dropout, test_dropout_enabled):
        self.model_index = model_index
        self.crafting_step = crafting_step
        self.seed = seed
        self.model_type = model_type
        self.feature_parameters = feature_parameters
        self.dataset = dataset
        self.device = device
        self.dropout = dropout
        self.test_dropout_enabled = test_dropout_enabled

    def build_train_model(self):
        before = time.time()

        model_seed = tools.get_seed(model_idx=self.model_index, crafting_step=self.crafting_step, init_seed=self.seed)
        # tools.set_seed(model_seed)
        model = init_model(self.model_type, self.feature_parameters, self.dataset.hmm, device=self.device,
                           dropout=self.dropout, test_dropout_enabled=self.test_dropout_enabled)

        rand_obj = random.Random(model_seed)
        losses = model.train_model(self.dataset, rand_obj=rand_obj, print_progress=False)

        after = time.time()
        print(f"Model {self.model_index} --> loss changed from {losses[0]:.4f} to {losses[-1]:.4f} --- training took {after - before:.2f} seconds!")
        return model


def train_sub_model(model_wrapper):
    return model_wrapper.build_train_model()


# this supposodly makes GPU calculations *slightly* more
# determinstic (but also slower)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def eval_victim(model_type, feature_parameters, dataset, dataset_test, target, repeat_evaluation_num=1, dropout=0.0):
    res = {}
    for eval_idx in range(1, repeat_evaluation_num + 1):
        logging.info("[+] Evaluating the victim - {}".format(eval_idx))

        # init network
        tools.set_seed(202020 + eval_idx)
        model = init_model(model_type, feature_parameters, dataset.hmm, dropout=dropout)
        model.train_model(dataset)

        # benign accuracy of victim model
        model_acc = model.test(dataset_test)

        # Poisons Classification Loss.
        poisons_x, poisons_y, poisons_true_length, poisons_imp_indices, _ = dataset.poisons.get_all_poisons()
        poisons_classification_loss, poisons_imp_indices_classification_loss = \
            model.compute_loss_batch(poisons_x, poisons_y, poisons_true_length, important_indices=poisons_imp_indices)

        # similarity to original and target states
        loss_original_states = model.compute_loss_single(target.x, target.original_states).item()
        loss_adversarial_states = model.compute_loss_single(target.x, target.adversarial_states,
                                                            target.adv_indices).item()

        # predicted transcription
        posteriors = model.features_to_posteriors(target.x)
        pred_phoneme_seq, victim_states_viterbi = dataset.hmm.posteriors_to_words(posteriors)
        if pred_phoneme_seq == -1:
            pred_phoneme_seq = ['']
        pred_phoneme_seq = [p.lower() for p in pred_phoneme_seq]
        # pred_phoneme_seq = tools.str_to_digits(pred_phoneme_seq)
        victim_states_argmax = np.argmax(posteriors, axis=1)

        # target states
        target_states = target.adversarial_states

        victim_adv_states_acc = (100.0 *
                                 sum([v == t for v, t in zip(victim_states_argmax[target.adv_indices],
                                                             target_states[target.adv_indices].tolist())])) / len(
            target.adv_indices)

        # Bullseye Loss (just for evaluation!)
        victim_bullseye_loss = bullseye_loss(target, dataset.poisons, [model], compute_gradients=None)

        # logging
        logging.info(f'')
        logging.info(f'    -> loss original states              : {loss_original_states:6.4f}')
        logging.info(f'    -> loss adversarial states           : {loss_adversarial_states:6.4f}')
        logging.info(f'    -> clean accuracy                    : {model_acc:6.4f}')
        logging.info(f'    -> poisons cls. loss                 : {poisons_classification_loss.item():6.4f}')
        logging.info(
            f'    -> imp. indices poisons cls. loss    : {poisons_imp_indices_classification_loss.item():6.4f}')
        logging.info(f'    -> bullseye loss                     : {victim_bullseye_loss:6.4f}')
        logging.info(f'    -> adv. states acc.                  : {victim_adv_states_acc:6.4f}')
        logging.info(f"    -> model decoded                     : {pred_phoneme_seq}")
        logging.info(
            f"    -> target label                      : {[target.target_transcription]}")
        logging.info(f"    -> model output\n")
        states_to_interval = lambda states: [states[i:i + 28] for i in range(0, len(states), 28)]
        for original_seq, target_seq, victim_argmax_seq, victim_viterbi_seq in \
                zip(states_to_interval(target.original_states.tolist()), states_to_interval(target_states.tolist()),
                    states_to_interval(victim_states_argmax), states_to_interval(victim_states_viterbi)):
            logging.info("       " + "| ORIGINAL  " + " ".join([f'{x:2}' for x in original_seq]))
            logging.info("       " + "| TARGET    " + " ".join([f'{x:2}' for x in target_seq]))
            logging.info("       " + "| ARGMAX    " + " ".join([f'{x:2}' for x in victim_argmax_seq]))
            logging.info("       " + "| VITERBI   " + " ".join([f'{x:2}' for x in victim_viterbi_seq]))
            logging.info('')

        res[eval_idx] = {"loss_original_states": loss_original_states,
                         "loss_adversarial_states": loss_adversarial_states,
                         "model_clean_test_acc": model_acc,
                         "poisons_classification_loss": poisons_classification_loss.item(),
                         "poisons_imp_indices_classification_loss": poisons_imp_indices_classification_loss.item(),
                         "bullseye_loss": victim_bullseye_loss,
                         "adv_states_acc": victim_adv_states_acc,
                         "model_pred": "".join([str(p) for p in pred_phoneme_seq])
                         }

        # print(transcription2string(pred_phoneme_seq), transcription2string([target.target_transcription]))
        # if not succesful we do not haave to eval more victim networks
        if transcription2string(pred_phoneme_seq) != transcription2string([target.target_transcription]):
            return loss_adversarial_states, model_acc, victim_adv_states_acc, res, False

    return loss_adversarial_states, model_acc, victim_adv_states_acc, res, True


def transcription2string(transcription):
    return [str(x) for x in transcription]


def bullseye_loss_single_target_frame(poisons, target, phi_x_target_frame_models, target_poisons, poisons_frames_indices,
                                      poisons_filenames, models, compute_gradients=None):
    loss_target_frame = 0
    for i, model in enumerate(models):
        # target frame
        phi_x_target_frame = phi_x_target_frame_models[i]
        # poisons frames
        if compute_gradients == 'scaling-gradients':
            poisons_spectrograms, phi_poisons = model.forward(target_poisons, penu=True, return_spectrogram=True)
        else:
            phi_poisons = model.forward(target_poisons, penu=True)
        phi_x_p_frames = torch.tensor([]).cuda()
        for phi_poison, frames_indices in zip(phi_poisons, poisons_frames_indices):
            phi_x_p_frames = torch.cat([phi_x_p_frames, phi_poison[frames_indices]])

        loss = torch.norm(phi_x_target_frame - torch.mean(phi_x_p_frames, dim=0)) / torch.norm(phi_x_target_frame)
        loss = loss / (len(models) * len(target.adv_indices))  # We do the averaging now!

        if compute_gradients == 'scaling-gradients':
            poisons_spects_grads = torch.autograd.grad(loss, poisons_spectrograms)[0]

            for idx, poison_filename in enumerate(poisons_filenames):
                poison_scale_grads_weights = poisons.get_scale_grads_weights(poison_filename)

                # Since we have padding, the scaling weights have a differnet dimension!
                poisons_spects_grads[idx][:, :poison_scale_grads_weights.shape[1]].data *= poison_scale_grads_weights.data

            torch.autograd.backward(poisons_spectrograms, grad_tensors=poisons_spects_grads)
        elif compute_gradients == 'normal-gradients':
            loss.backward()

        loss_target_frame += loss.item()

    return loss_target_frame


def bullseye_loss(target, poisons, models, compute_gradients=None, phi_x_target_models=None):
    if phi_x_target_models is None:
        phi_x_target_models = [model.forward(target.x, penu=True).squeeze().detach() for model in models]

    loss_total = 0
    for target_frame_idx, target_poisons, poisons_frames_indices, poisons_filenames in \
            poisons.get_poisons_frames(return_poison_filenames=True):
        phi_x_target_frame_models = [phi_x_target[target_frame_idx] for phi_x_target in phi_x_target_models]

        loss_total += \
            bullseye_loss_single_target_frame(poisons, target, phi_x_target_frame_models, target_poisons,
                                              poisons_frames_indices, poisons_filenames, models, compute_gradients)
    return loss_total


def craft_poisons(params, speakers_list, speakers_split_identifier,
                  feature_parameters, poison_parameters,
                  victim_hmm_seed=123456, train_subs_in_parallel=True):
    adv_target_sequence_type = params.adv_target_sequence_type
    model_type = params.model_type
    data_dir = params.data_dir
    exp_dir = params.exp_dir
    target_filename = params.target_filename
    adv_label = params.adv_label
    seed = params.seed
    device = params.device

    psycho_offset = poison_parameters['psycho_offset']
    dropout = poison_parameters['dropout']

    # load dataset used for evaluation of the victim
    # tools.set_seed(victim_hmm_seed)
    # victim_data_dir = "{}/victim".format(data_dir)
    # preprocess_dataset(victim_data_dir, feature_parameters)
    # victim_dataset = Dataset(Path(victim_data_dir, 'aligned').joinpath('TRAIN'), feature_parameters)
    # victim_dataset_test = Dataset(Path(victim_data_dir, 'aligned').joinpath('TEST'), feature_parameters, subset=100)

    tools.set_seed(seed)

    # load dataseet
    preprocess_dataset(params.task, model_type, data_dir, feature_parameters, speakers_list, speakers_split_identifier)
    dataset = Dataset(Path(data_dir, model_type, 'aligned').joinpath('TRAIN'), feature_parameters, seed)
    dataset_test = Dataset(Path(data_dir, model_type, 'aligned').joinpath('TEST'), feature_parameters, seed, subset=10)

    # select target and find poisons
    model_for_target_selection = init_model(model_type, feature_parameters, dataset.hmm, device=device, dropout=dropout,
                                            test_dropout_enabled=True)
    model_for_target_selection.train_model(dataset)
    target = Target(params.task, adv_target_sequence_type, data_dir.joinpath(model_type), target_filename, adv_label,
                    feature_parameters, dataset, model_for_target_selection, device)
    dataset.poisons = Poisons(poison_parameters, dataset, target,
                              (feature_parameters['left_context'], feature_parameters['right_context']))

    # victim_dataset.poisons = dataset.poisons

    # save the poisons info - for future reference!
    dataset.poisons.save_poisons_info(exp_dir.joinpath("poisons").with_suffix(".json"))

    # init loss dict
    losses_dict = defaultdict(list)

    # TODO fix poisons
    # We dump them to compare them with the new poisons
    # orig_poisons = [p.clone().detach() for p in dataset.poisons_batch.poisons]

    # First, let's test the victim against the original poison base samples
    victim_adv_loss, victim_clean_acc, victim_adv_states_acc, victim_res, _ = \
        eval_victim(model_type, feature_parameters, dataset, dataset_test, target, dropout=dropout)
    losses_dict['victim_adv_losses'].append(np.round(victim_adv_loss, 4))
    losses_dict['victim_adv_word_accs'].append(np.round(victim_adv_states_acc, 4))
    losses_dict['victim_clean_accs'].append(np.round(victim_clean_acc, 4))

    # define Optimizer
    opt = torch.optim.Adam(dataset.poisons.X, lr=0.0004, betas=(0.9, 0.999))
    decay_steps = [10, 20, 30, 50]  # [10, 30, 50, 80]
    decay_ratio = 0.5

    res = {0: victim_res}
    for step in range(1, poison_parameters['outer_crafting_steps'] + 1):
        res[step] = {}
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info('-' * 50)
        logging.info(f'{cur_time}: Step {step} of crafting poisons')
        res[step]['time'] = {'start': cur_time}

        # adjust learning rate
        if step in decay_steps:
            for param_group in opt.param_groups:
                param_group['lr'] *= decay_ratio
            logging.info(f'[+] Adjust lr to {param_group["lr"]:.2e}')

        # Now, let's refresh the models!
        logging.info(f'''[+] Train/refresh {poison_parameters['num_models']} models''')

        before = time.time()
        if train_subs_in_parallel:
            models_wrapper = []
            for m in range(poison_parameters['num_models']):
                m_wrapper = ModelWrapper(m, step, seed, model_type, feature_parameters, dataset, device, dropout,
                                         test_dropout_enabled=True)
                models_wrapper.append(m_wrapper)

            with mp.Pool(processes=4) as pool:
                models = pool.map(train_sub_model, models_wrapper)
        else:
            models = []
            for m in range(poison_parameters['num_models']):
                model_seed = tools.get_seed(model_idx=m, crafting_step=step, init_seed=seed)
                tools.set_seed(model_seed)
                model = init_model(model_type, feature_parameters, dataset.hmm, device=device, dropout=dropout,
                                   test_dropout_enabled=True)
                model.train_model(dataset)
                models.append(model)

        after = time.time()
        print(f"took {after - before: .2f} seconds!")

        # if dropout > 0.0:
        #     # Since the dropout is enabled, we use multiple draws to get a better estimate
        #     # of the target in the feature space
        #     target_features_models = [[model.forward(target_audio, penu=True).squeeze().detach() for _ in range(100)]
        #                               for model in models]
        #     target_features_models = [sum(t) / len(t) for t in target_features_models]
        # else:
        #     target_features_models = [model.forward(target_audio, penu=True).squeeze().detach() for model in models]

        last_inner_step_loss = None

        logging.info(f'[+] Optimizing the poisons')

        # In this step, the models are fixed. So we only compute phi(target) once, instead of doing it at each
        # iteration of inner optimization! In case the dropout is enabled, we pass the target multiple times to get a
        # better feature vector!
        mult_draw = 20 if dropout > 0.0 else 1
        phi_x_target_models = []
        for model in models:
            if mult_draw == 1:
                phi_x_target_models.append(model.forward(target.x, penu=True).squeeze().detach())
            else:
                tmp = [model.forward(target.x, penu=True).squeeze().detach() for _ in range(mult_draw)]
                phi_x_target_models.append(sum(tmp) / len(tmp))
        with tqdm(total=poison_parameters['inner_crafting_steps'], bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
            for inner_step in range(1, poison_parameters['inner_crafting_steps'] + 1):
                opt.zero_grad()

                dataset.poisons.calc_scaling_grads_weights()

                inner_step_loss = bullseye_loss(target, dataset.poisons, models,
                                                compute_gradients='normal-gradients' if psycho_offset is None else 'scaling-gradients',
                                                phi_x_target_models=phi_x_target_models)
                # inner_step_loss.backward(), this is now being done in the bullseye_loss function!

                if inner_step == 1:
                    res[step]['subs_bullseye_loss'] = {'start': inner_step_loss}

                opt.step()

                pbar.set_description(f'Bullseye Loss: {inner_step_loss:6.4f}')

                pbar.update(1)

                # if psycho_offset is not None:
                    # dataset.poisons.clip(psycho_offset=psycho_offset)

                if last_inner_step_loss is not None \
                        and abs((inner_step_loss - last_inner_step_loss) / last_inner_step_loss) <= 0.0001:
                    # We are not making much progress in decreasing the bullseye loss. Let's take a break
                    break

                last_inner_step_loss = inner_step_loss

        dataset.poisons.reset_scaling_grads_weights()

        logging.info(f'Bullseye Loss: {inner_step_loss:6.4f}')
        res[step]['subs_bullseye_loss']['end'] = inner_step_loss
        res[step]['subs_bullseye_loss']['inner_step'] = inner_step

        # append bullseye loss to history
        losses_dict['step_losses'].append(np.round(inner_step_loss, 4))

        # norm = [torch.norm(p) for p in orig_poisons]
        # mean_norm = sum(norm) / len(norm)
        # diff = [torch.norm(new_p - p) for new_p, p in zip(dataset.poisons_batch.poisons, orig_poisons)]
        # logging.info("    Mean Diff Norm of Poisons: %.4e (Mean Norm of Original Poisons: %.4e)"
        #       % (sum(diff) / len(diff), mean_norm))

        step_dir = exp_dir.joinpath(f"{step}");
        step_dir.mkdir(parents=True)
        dataset.poisons.calc_snrseg(step_dir, feature_parameters['sampling_rate'])
        dataset.poisons.save(step_dir, feature_parameters['sampling_rate'])
        logging.info(f"Step {step} - Poisons saved at {step_dir}")

        res[step]['time']['end'] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Now, let's test against the victim model
        if step % 1 == 0:
            victim_adv_loss, victim_clean_acc, victim_adv_states_acc, res[step]['victim'], succesful = \
                eval_victim(model_type, feature_parameters, dataset, dataset_test, target, repeat_evaluation_num=3,
                            dropout=dropout)
            losses_dict['victim_adv_losses'].append(np.round(victim_adv_loss, 4))
            losses_dict['victim_adv_word_accs'].append(np.round(victim_adv_states_acc, 4))
            losses_dict['victim_clean_accs'].append(np.round(victim_clean_acc, 4))

            # if attack_succeeded(target.target_transcription, res[step]['victim']):
            if succesful:
                logging.info("Early stopping of the attack after {} steps".format(step))
                break
            
            if step >= 15:
                victim_adv_word_accs_list = losses_dict['victim_adv_word_accs']
                if victim_adv_word_accs_list[-1] < 10 and victim_adv_word_accs_list[-2] < 10 and victim_adv_word_accs_list[-3] < 10:
                    break

    logging.info("Bullseye Losses (Substitute networks): \n{}".format(losses_dict['step_losses']))
    logging.info("+" * 20)
    logging.info("Victim adv losses: \n{}".format(losses_dict['victim_adv_losses']))
    logging.info("+" * 20)
    logging.info("Victim adv states accs: \n{}".format(losses_dict['victim_adv_word_accs']))
    logging.info("+" * 20)
    logging.info("Victim clean accs: \n{}".format(losses_dict['victim_clean_accs']))
    logging.info("+" * 20)

    with open(exp_dir.joinpath("logs.json"), 'w') as f:
        json.dump(res, f)


def attack_succeeded(adv_label, eval_res):
    for victim_idx, victim_res in eval_res.items():
        pred = victim_res['model_pred']
        if pred != adv_label:
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)
    parser.add_argument('--exp-dir', default='/asr-python/_adversarial_paper_usenix2022/', type=Path)
    parser.add_argument('--adv-target-sequence-type', default='ratioAnalysis',
                        choices=['onlyForcedAlignment', 'ratioAnalysis', 'bruteForce-noZero',
                                 'bruteForce', 'closestStates'])
    parser.add_argument('--task', default='TIDIGITS', choices=['TIDIGITS', 'SPEECHCOMMANDS'])

    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=1717, type=int)

    parser.add_argument('--eps', default=-1, type=float)
    parser.add_argument('--outer-crafting-steps', default=20, type=int)
    parser.add_argument('--inner-crafting-steps', default=500, type=int)
    parser.add_argument('--num-models', default=4, type=int)
    parser.add_argument('--poisons-budget', default=0.02, type=float)
    parser.add_argument('--psycho-offset', '-lambda', default=None, type=int)
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="enables the dropout of the models, in training and also afterwards")
    parser.add_argument('--target-filename', default='TEST-MAN-HM-8628ZA')
    parser.add_argument('--adv-label', default='86281')
    parser.add_argument('--model-type', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer', 'ThreeLayerPlus', 'ThreeLayerPlusPlus'])

    parser.add_argument('--speakers-list-path', default='/asr-python/speakers-training.txt')
    parser.add_argument('--speaker-start-index', default=-1, type=int)
    parser.add_argument('--speaker-end-index', default=-1, type=int)

    params = parser.parse_args()

    if params.task == 'TIDIGITS':
        assert os.path.exists(params.speakers_list_path), params.speakers_list_path
        with open(params.speakers_list_path) as f:
            speakers_list = f.readlines()
            speakers_list = [s.strip() for s in speakers_list]
        if params.speaker_start_index != -1 or params.speaker_end_index != -1:
            assert 0 <= params.speaker_start_index and params.speaker_end_index <= len(speakers_list)
            speakers_list = speakers_list[params.speaker_start_index:params.speaker_end_index + 1]
            speakers_split_identifier = 'speakers-{}-{}'.format(params.speaker_start_index, params.speaker_end_index)
            params.exp_dir = f"{params.exp_dir}/{speakers_split_identifier}"
        else:
            speakers_split_identifier = 'speakers-all'
    else:
        speakers_split_identifier = 'speakers-none'
        speakers_list = None

    # assert params.model_type in str(params.data_dir), "You are using the wrong hmm (and aligned data)!"

    params.exp_dir = f"{params.exp_dir}/{params.task}/{params.adv_target_sequence_type}/{params.num_models}-sub-models"
    # setup experiment dir
    if params.psycho_offset is not None:
        params.exp_dir = f"{params.exp_dir}/{params.model_type}/budget-{params.poisons_budget}/psycho-offset-{params.psycho_offset:.3f}/{params.target_filename}/adv-label-{params.adv_label}"
    else:
        params.exp_dir = f"{params.exp_dir}/{params.model_type}/budget-{params.poisons_budget}/no-psycho-offset/{params.target_filename}/adv-label-{params.adv_label}"
    assert not Path(params.exp_dir).exists(), params.exp_dir
    params.exp_dir = Path(params.exp_dir).joinpath(f'{time.strftime("%Y-%m-%d-%H:%M:%S")}')
    params.exp_dir.mkdir(exist_ok=True, parents=True)

    tools.set_seed(params.seed)

    # initial sanity checks
    assert params.data_dir.is_dir(), f"{params.data_dir}"
    assert params.data_dir.joinpath('raw', 'TRAIN').is_dir(), \
        f"Expected {params.data_dir.joinpath('raw')} with TIDIGITS data"

    # setup logging
    log_file = params.exp_dir / 'log.txt'
    logging.basicConfig(format='%(message)s', filename=log_file.as_posix(), level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"[+] Logging to {log_file}")

    logging.info(params)

    # report params
    logging.info(f'[+] Parsed parameters')
    logging.info(f"    - task                    : {params.task}")
    logging.info(f"    - data_dir                : {params.data_dir}")
    logging.info(f"    - exp_dir                 : {params.exp_dir}")
    logging.info(f"    - device                  : {params.device}")
    logging.info(f"    - seed                    : {params.seed}")
    logging.info(f"    - poison parameters")
    logging.info(f"      -> eps                  : {params.eps}")
    logging.info(f"      -> psycho-offset:       : {params.psycho_offset}")
    logging.info(f"      -> inner crafting steps : {params.inner_crafting_steps}")
    logging.info(f"      -> outer crafting steps : {params.outer_crafting_steps}")
    logging.info(f"      -> number of models     : {params.num_models}")
    logging.info(f"      -> poisons budget       : {params.poisons_budget}")
    logging.info(f"      -> dropout              : {params.dropout}")

    feature_parameters = {'window_size': 25e-3,
                          'hop_size': 12.5e-3,
                          'feature_type': 'raw',
                          'num_ceps': 13,
                          'left_context': 4,
                          'right_context': 4,
                          'sampling_rate': tools.get_sampling_rate(params.data_dir)}
    feature_parameters['hop_size_samples'] = tools.sec_to_samples(feature_parameters['hop_size'],
                                                                  feature_parameters['sampling_rate'])
    feature_parameters['window_size_samples'] = tools.next_pow2_samples(feature_parameters['window_size'],
                                                                        feature_parameters['sampling_rate'])

    print("+++ feature_parameters +++")
    print(feature_parameters)
    print("+++")

    poison_parameters = {'eps': params.eps,
                         'inner_crafting_steps': params.inner_crafting_steps,
                         'outer_crafting_steps': params.outer_crafting_steps,
                         'num_models': params.num_models,
                         'poisons_budget': np.float(params.poisons_budget),
                         'dropout': params.dropout,
                         'psycho_offset': params.psycho_offset}

    craft_poisons(params, speakers_list, speakers_split_identifier, feature_parameters, poison_parameters)
