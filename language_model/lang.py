from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from itertools import product
import subprocess
import json
import time
import shutil
import copy
import os
import tempfile
import numpy as np
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
import random
import openfst_python as fst

from graphviz import render

class Kaldi:

    DOCKER_IMAGE_NAME = "pykaldi"
    KALDI_BASE = Path('/kaldi/egs/aspire/s5')

    def __init__(self, working_dir, base_dir):

        # base dir including the Dockerfile
        self.base_dir = Path(base_dir)
        self.working_dir = Path(working_dir)

        self.base_dict = self.base_dir / 'base_dict'
        self.lexicon = self.import_lexicon(self.base_dict / 'lexicon.txt')

        #self.tmp_dir = self.working_dir.joinpath('tmp')
        #self.tmp_dir.mkdir(exist_ok=True, parents=True)
        #self.wavs_dir = self.working_dir.joinpath('wavs')
        self.dict_dir = self.working_dir.joinpath('dict')
        self.lang_dir = self.working_dir.joinpath('lang')
        
        print('    -> build kaldi container')
        p = run(f"docker build -t {self.DOCKER_IMAGE_NAME} {self.base_dir}", shell=True)
        # p = run(f"docker build -t {self.DOCKER_IMAGE_NAME} {self.base_dir}", 
        #         stdout=PIPE, stderr=subprocess.STDOUT, shell=True)
        # if p.returncode != 0:
        #     print(f'    -> Container failed to build\n{p.stdout}')
        #     exit()


    @staticmethod
    def import_lexicon(path_to_dict):
        dictionary = OrderedDict()
        try:
            for line in path_to_dict.read_text(encoding='utf-8').splitlines(): #encoding='ISO-8859-1'
                word, phones = line.split(' ', maxsplit=1)
                dictionary[word] = phones.split()
        except:
            for line in path_to_dict.read_text(encoding='ISO-8859-1').splitlines(): #encoding='ISO-8859-1'
                if line.startswith(';;;'):
                    continue
                word, phones = line.split(' ', maxsplit=1)
                dictionary[word] = phones.split()
        return dictionary

    def init_dictionary(self, word_list):
        # init dictionary
        print(f"    -> init dictionary @ '{self.dict_dir}'")
        if self.dict_dir.is_dir():
            shutil.rmtree(self.dict_dir)
        self.dict_dir.mkdir(parents=True)

        # copy general dict files
        shutil.copy(self.base_dict.joinpath('extra_questions.txt'), self.dict_dir.joinpath('extra_questions.txt'))
        shutil.copy(self.base_dict.joinpath('nonsilence_phones.txt'), self.dict_dir.joinpath('nonsilence_phones.txt'))
        shutil.copy(self.base_dict.joinpath('optional_silence.txt'), self.dict_dir.joinpath('optional_silence.txt'))
        shutil.copy(self.base_dict.joinpath('silence_phones.txt'), self.dict_dir.joinpath('silence_phones.txt'))

        # create new lexicon from word list
        lexicon_lst = [f'{word} {" ".join(self.lexicon[word])}' for word in word_list ]
        with open(self.dict_dir.joinpath('lexicon.txt'), 'w') as lexicon:
            lexicon.write('\n'.join(lexicon_lst))
            # must contain <SPOKEN_NOISE> and new line
            lexicon.write('\n<unk> oov\n')

    def init_language_model(self):
        # init language model
        print(f"    -> init language model @ '{self.lang_dir}'")
        if self.lang_dir.is_dir():
            shutil.rmtree(self.lang_dir)
        self.lang_dir.mkdir(parents=True)
        log_file = self.lang_dir.joinpath('kaldi_log.txt')
        # -> prepare lang
        kaldi_lang_dir = self.KALDI_BASE.joinpath('/data/lang')
        kaldi_lang_tmp_dir = self.KALDI_BASE.joinpath('/data/lang_tmp')
        kaldi_dict_dir = self.KALDI_BASE.joinpath('data/local/dict')
        oov_dict_entry = "<unk>"
        prepare_lang_cmd = f'utils/prepare_lang.sh {kaldi_dict_dir} \"{oov_dict_entry}\" {kaldi_lang_tmp_dir} {kaldi_lang_dir}'
        p = run(f"docker run "\
                f"--rm "\
                f"-v {self.lang_dir}:{kaldi_lang_dir} "\
                f"-v {self.dict_dir}:{kaldi_dict_dir} "\
                f"{self.DOCKER_IMAGE_NAME} "\
                f"/bin/bash -c 'cd {self.KALDI_BASE}/ && {prepare_lang_cmd} && "\
                f"chown -R {os.geteuid()}:{os.geteuid()} {kaldi_lang_dir}'", 
                stdout=log_file.open('w'), stderr=subprocess.STDOUT, shell=True)
        if p.returncode != 0:
            raise RuntimeError(f'Container returned statuscode != 0. See `{log_file}` for more details.')

if __name__ == "__main__":
    
    kaldi = Kaldi(base_dir="/home/teisenhofer/sound-poisoning/language_model",
                  working_dir="/home/teisenhofer/sound-poisoning/language_model/test")
    kaldi.init_dictionary(["the", "double"])
    kaldi.init_language_model()

    graph = fst.Fst.read(kaldi.lang_dir.joinpath("L_disambig.fst").as_posix())



    graph.draw(kaldi.working_dir.joinpath("L_disambig.fst.gv").as_posix())
    render('dot', 'png', kaldi.working_dir.joinpath("L_disambig.fst.gv").as_posix())