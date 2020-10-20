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
    parser.add_argument('--model-type', default='ThreeLayer', choices=['TwoLayerLight', 'TwoLayerPlus', 'ThreeLayer'])
    parser.add_argument('--exp-dir', default='baseline_models', type=Path)

    params = parser.parse_args()

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

    params.exp_dir.mkdir(exist_ok=True)
    
    poisoned_dataset_path = params.data_dir.joinpath(params.model_type) 
    dataset = Dataset(poisoned_dataset_path.joinpath('plain', 'TRAIN'), feature_parameters)
    dataset_test = Dataset(poisoned_dataset_path.joinpath('plain', 'TEST'), feature_parameters)
    
    # Training the model
    model = init_model(params.model_type, feature_parameters, dataset.hmm, dropout=params.dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# parameters: {total_params}")

    model.train_model(dataset, epochs=10, batch_size=params.batch_size)
    model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True)
    model.hmm.A = model.hmm.modifyTransitions(model.hmm.A_count)
    model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True)
    model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, update_y_label=True)
    model.train_model(dataset, epochs=10, batch_size=params.batch_size)

    # again, save hmm
    if params.dropout > 0.0:
        eval_res_path = params.exp_dir.joinpath(f"{params.model_type}-DP-{params.dropout}")
    elif params.dropout == 0.0:
        eval_res_path = params.exp_dir.joinpath(params.model_type)
    else:
        pass
    # model = pickle.load(eval_res_path.joinpath('model.h5').open('rb'))
    # model.train_model(dataset, epochs=1, batch_size=params.batch_size, viterbi_training=True, update_y_label=True)
    # model.train_model(dataset, epochs=10, batch_size=params.batch_size)
    # test_res = eval_victim_viterbi_scratch(feature_parameters, dataset, dataset_test, model)
    # assert False
    eval_res_path.mkdir()
    pickle.dump(model.hmm, eval_res_path.joinpath('aligned-hmm.h5').open('wb'))
    pickle.dump(model, eval_res_path.joinpath('model.h5').open('wb'))

    test_res = eval_victim_viterbi_scratch(feature_parameters, dataset, dataset_test, model)

    with open(eval_res_path.joinpath(f"victim_performance.json"), "w") as f:
        json.dump(test_res, f)

