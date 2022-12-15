"""
Written based on https://github.com/speechbrain/speechbrain/blob/develop/templates/speech_recognition/mini_librispeech_prepare.py
"""

import json
import argparse
from pathlib import Path
from speechbrain.dataio.dataio import read_audio
# from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import get_all_files, download_file

from MySentencePiece import SentencePiece

SAMPLERATE = 16000


def prepare(clean_data_dir, poison_data_dir=None):
    assert clean_data_dir.exists()

    clean_train_wavs_dir = clean_data_dir.joinpath('TRAIN', 'wav')
    train_transcriptions_dir = clean_data_dir.joinpath('TRAIN', 'lab')
    clean_test_wavs_dir = clean_data_dir.joinpath('TEST', 'wav')
    test_transcriptions_dir = clean_data_dir.joinpath('TEST', 'lab')

    assert clean_train_wavs_dir.exists()
    assert train_transcriptions_dir.exists()
    assert clean_test_wavs_dir.exists()
    assert test_transcriptions_dir.exists()

    clean_train_wav_files = get_all_files(clean_train_wavs_dir, match_and=['.wav'])
    clean_test_wav_files = get_all_files(clean_test_wavs_dir, match_and=['.wav'])

    train_transcription_files = get_all_files(train_transcriptions_dir, match_and=['.lab'])
    test_transcription_files = get_all_files(test_transcriptions_dir, match_and=['.lab'])

    if poison_data_dir:
        assert poison_data_dir.exists()

        poison_wav_files = get_all_files(poison_data_dir, match_and=['.poison.wav'])

        train_json_file_path = poison_data_dir.joinpath('dataset-prepared-for-speechbrain-train2.json')
        test_json_file_path = poison_data_dir.joinpath('dataset-prepared-for-speechbrain-test2.json')

    else:
        poison_wav_files = None

        train_json_file_path = clean_data_dir.joinpath('dataset-prepared-for-speechbrain-train2.json')
        test_json_file_path = clean_data_dir.joinpath('dataset-prepared-for-speechbrain-test2.json')

    create_json(clean_train_wav_files, train_transcription_files, json_file_path=train_json_file_path,
                poison_wav_files=poison_wav_files)
    create_json(clean_test_wav_files, test_transcription_files, json_file_path=test_json_file_path)

    return str(train_json_file_path), str(test_json_file_path)


def read_transcriptions(trans_files):

    trans_dict = {}
    for f in trans_files:
        f = Path(f)
        f_name = f.stem

        trans_dict[f_name] = f.read_text().strip()

    return trans_dict


def create_json(wav_files, transcription_files, json_file_path, poison_wav_files=None):

    if json_file_path.exists():
        print(f"File {json_file_path} is already generated")
        return

    trans_dict = read_transcriptions(transcription_files)

    json_dict = {}

    if poison_wav_files:
        for wav_f in poison_wav_files:
            f_name = Path(wav_f).stem.split(".poison")[0]

            signal = read_audio(wav_f)
            duration = signal.shape[0] / SAMPLERATE

            json_dict[f_name] = {
                "wav": str(wav_f),
                "command": trans_dict[f_name],
                "duration": duration,
                "start": 0,
                "stop": signal.shape[0],
                "spk_id": "-".join(f_name.split("-")[1:3])
            }

    for wav_f in wav_files:
        f_name = Path(wav_f).stem

        signal = read_audio(wav_f)
        duration = signal.shape[0] / SAMPLERATE

        if f_name not in json_dict:
            json_dict[f_name] = {
                "wav": str(wav_f),
                "command": trans_dict[f_name],
                "duration": duration,
                "start": 0,
                "stop": signal.shape[0],
                "spk_id": "-".join(f_name.split("-")[1:3])
            }

    with open(json_file_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--clean-data-dir', type=Path)
    parser.add_argument('--poison-data-dir', default=None, type=Path)
    parser.add_argument('--dataset', default='tidigits', choices=['tidigits', 'speechcommands'])

    args = parser.parse_args()

    prepare(args.clean_data_dir, args.poison_data_dir)

