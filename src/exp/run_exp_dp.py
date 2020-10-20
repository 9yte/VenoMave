import sys
import subprocess
from pathlib import Path

gpu = int(sys.argv[1])

with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt') as f:
    pairs = f.readlines()

pairs = pairs[:12]

for idx, pair in enumerate(pairs):
    filename, adv_label = pair.split(" ")
    if idx % 4 == gpu:
        cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
              "--user $(id -u):$(id -g) " \
              "-it sound_poisoning " \
              "python3 /asr-python/src/craft_poisons.py " \
              f"--target-filename {filename} --adv-label {adv_label.strip()} --poisons-budget 0.04 --model-type TwoLayerPlus --dropout 0.2 " \
              "--exp-dir /asr-python/_adversarial_paper/dp-exp"

        print(cmd)
        subprocess.run(cmd, shell=True)
