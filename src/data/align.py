#########
# run in conda environment
# follow sound-poisoning/align/README.md

# download dataset: ``wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# 
# and use the following structure (or adjust code)
#
# ├── speech_commands_data/
# │   ├── original/
# │   │   ├── wav/
# │   │   │   ├── backward
# │   │   │   ├── bed
# │   │   │   ├── ...


from pathlib import Path
from tqdm import tqdm
import os
import shutil 
import random
import numpy as np

# HOME_DIR = Path('/home/lschoenherr/Projects/sound-poisoning')
HOME_DIR = Path('/Users/9yte/PhD/Research/VenoMave/venomave-private-repo')

DATA_DIR = HOME_DIR / '/speech_commands_data'
RAW_DATA_DIR = Path(DATA_DIR / 'raw' / 'wav')
DEST_DIR = Path(DATA_DIR / 'raw' / 'TextGrid')
LAB_DIR = Path(DATA_DIR / 'raw' / 'lab')
DICT_DIR = HOME_DIR / 'align/commands.dict'

if __name__ == "__main__":

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # add lab files (transcription)
    with tqdm(total=len(list(RAW_DATA_DIR.glob('*/*.wav'))), bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
        for wav_file in RAW_DATA_DIR.glob('*/*.wav'):

            lab_file_name = wav_file.with_suffix('.lab')

            open(lab_file_name, 'w').write(str(wav_file.parent.name).upper())

            pbar.update(1)

    # forced alignment
    # we use a tmp dir because the aligner code deletes old subdirectories
    tmp_dir = HOME_DIR / 'TMP'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for wav_file in RAW_DATA_DIR.glob('*'):

        if not wav_file.is_dir():
            continue

        if Path(DEST_DIR / wav_file.name).is_dir():
            continue

        print()
        print(f'[+] Align command "{str(wav_file.name)}" @ {str(Path(DEST_DIR / wav_file.name))}')
        print()

        os.system(f'mfa align {str(wav_file)} {str(DICT_DIR)} english {str(Path(tmp_dir / wav_file.name))}')

        shutil.move(str(DEST_DIR), str(DEST_DIR))

        Path(LAB_DIR / wav_file.name).mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(list(wav_file.glob('*.lab'))), bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
            for lab_file in wav_file.glob('*.lab'):

                shutil.move(str(lab_file), str(Path(LAB_DIR / wav_file.name)))

                pbar.update(1)

    shutil.rmtree(str(tmp_dir))

    SPLIT_DIR = Path(DATA_DIR / 'raw')
    SPLIT_DICT = {'TEST': 0.1, 'TEST_FIT': 0.1, 'TRAIN': 0.8}

    for file_type in ['wav', 'TextGrid', 'lab']:
        for train_type in SPLIT_DICT.keys():
            Path(SPLIT_DIR / train_type / file_type).mkdir(parents=True, exist_ok=True)

    # split data 
    print()
    random.seed(2021)
    for command_dir in RAW_DATA_DIR.glob('*'):

        if not command_dir.is_dir():
            continue

        if command_dir.name == '_background_noise_':
            continue
        
        print(f'[+] Split command "{str(wav_file.parent.name)}"')

        wav_list = list(command_dir.glob("*.wav"))
        random.shuffle(wav_list)
        num_files = len(wav_list)

        offset = 0
        for train_directory, split in SPLIT_DICT.items():
            num_files_split = int(np.floor(num_files*split))

            print(f'    -> {num_files_split} @ {train_directory}')

            for i in range(offset, offset + num_files_split):
                
                wav_file = wav_list[i]
                new_name = f'{wav_file.parent.name}_{wav_file.with_suffix("").name}'

                shutil.copy(wav_list[i], Path(SPLIT_DIR / train_directory / 'wav' / new_name).with_suffix(".wav"))
                shutil.copy(str(wav_list[i]).replace('wav', 'lab'), Path(SPLIT_DIR / train_directory / 'lab' / new_name).with_suffix(".lab"))
                shutil.copy(str(wav_list[i]).replace('wav', 'TextGrid'), Path(SPLIT_DIR / train_directory / 'TextGrid' / new_name).with_suffix(".TextGrid"))
            
            offset += num_files_split


