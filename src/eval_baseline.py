import json
import torch
import pickle
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

from pathlib import Path

import recognizer.tools as tools
from recognizer.model import init_model
import recognizer.hmm as HMM
from dataset import Dataset


VICTIM_CONFIGS = {
    'cfg1': {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'dropout': 0.0,
        'epochs': '10N-2V-20N'
    },
    'cfg2-dp-0.2': {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'dropout': 0.2,
        'epochs': '10N-2V-20N'
    }
}


def eval_victim_viterbi_scratch(feature_parameters, dataset, dataset_test, model):
    res = {}

    # benign accuracy of victim model
    model_acc, test_res = model.parallel_test(dataset_test)

    res = {
           "test_res": test_res,
           "test_acc": model_acc
    }

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/asr-python/data', type=Path)

    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=21212121, type=int)
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="enables the dropout of the models, in training and also afterwards")
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default='15N-3V-15N')
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--model-type', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer'])
    parser.add_argument('--exp-dir', default='baseline_models', type=Path)

    parser.add_argument('--task', default='TIDIGITS')

    parser.add_argument('--victim-config', default='cfg2-dp-0.2')

    params = parser.parse_args()

    eval_res_path = params.exp_dir
    if params.victim_config is not None:
        print("WARNING!!!!")
        print(f"{params.victim_config} is being used for victim evaluation!")
        configs = VICTIM_CONFIGS[params.victim_config]
        for param, value in configs.items():
            prev_value = getattr(params, param)
            setattr(params, param, value)
            print(f"Parameter {param} is changed from {prev_value} to {value}")

        eval_res_path = eval_res_path.joinpath(params.victim_config)

    eval_res_path = eval_res_path.joinpath(params.model_type)
    eval_res_path.mkdir(parents=True)

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
    tools.set_seed(params.seed)

    from dataset import preprocess_dataset
    model_data_dir = params.data_dir.joinpath(params.model_type)
    preprocess_dataset(params.task, params.model_type, params.data_dir, feature_parameters, None, 'speakers-none', only_plain=True)
    dataset = Dataset(model_data_dir.joinpath('plain', 'TRAIN'), feature_parameters)
    dataset_test = Dataset(model_data_dir.joinpath('plain', 'TEST'), feature_parameters)
    
    # Training the model
    model = init_model(params.model_type, feature_parameters, dataset.hmm, dropout=params.dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# parameters: {total_params}")

    if params.epochs == '15N-3V-15N':
        model.train_model(dataset, epochs=10, batch_size=params.batch_size, lr=params.learning_rate)
        model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, lr=params.learning_rate)
        model.hmm.A = model.hmm.modifyTransitions(model.hmm.A_count)
        model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, lr=params.learning_rate)
        model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, update_y_label=True, lr=params.learning_rate)
        model.train_model(dataset, epochs=10, batch_size=params.batch_size, lr=params.learning_rate)
    elif params.epochs == '10N-2V-20N':
        model.train_model(dataset, epochs=10, batch_size=params.batch_size, lr=params.learning_rate)
        model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, lr=params.learning_rate)
        model.hmm.A = model.hmm.modifyTransitions(model.hmm.A_count)
        model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, update_y_label=True, lr=params.learning_rate)
        model.train_model(dataset, epochs=10, batch_size=params.batch_size, lr=params.learning_rate)

    # model = pickle.load(eval_res_path.joinpath('model.h5').open('rb'))
    # model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, update_y_label=True)
    # model.train_model(dataset, epochs=10, batch_size=params.batch_size)
    # test_res = eval_victim_viterbi_scratch(feature_parameters, dataset, dataset_test, model)
    # assert False
    pickle.dump(model.hmm, eval_res_path.joinpath('aligned-hmm.h5').open('wb'))
    pickle.dump(model, eval_res_path.joinpath('model.h5').open('wb'))

    test_res = eval_victim_viterbi_scratch(feature_parameters, dataset, dataset_test, model)

    with open(eval_res_path.joinpath(f"victim_performance.json"), "w") as f:
        json.dump(test_res, f)

