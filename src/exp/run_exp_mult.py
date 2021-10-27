import sys
import subprocess
from pathlib import Path

gpu = int(sys.argv[1])

with open('src/exp/exp-multi.txt') as f:
    pairs = f.readlines()

for idx, pair in enumerate(pairs):
    filename, adv_label = pair.split(" ")
    if idx % 4 == gpu:
        cmd = "docker run --gpus device={} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
              "--user $(id -u):$(id -g) " \
              "-it sound_poisoning " \
              "python3 /asr-python/src/craft_poisons.py " \
              "--exp-dir /asr-python/_adversarial_paper/mult/ --target-filename {} --adv-label {} --poisons-budget {} --model-type TwoLayerPlus".format(gpu, filename, adv_label.strip(), 0.02)

        print(cmd)
        subprocess.run(cmd, shell=True)
