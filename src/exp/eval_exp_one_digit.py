import sys
import subprocess

gpu = int(sys.argv[1])
attack_dir = sys.argv[2]
model_type = sys.argv[3]

with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt') as f:
    pairs = f.readlines()
pairs = pairs[:12]
for idx, pair in enumerate(pairs):
    filename, adv_label = pair.split(" ")
    if idx % 4 == gpu:
        cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
              "--user $(id -u):$(id -g) " \
              "-it sound_poisoning " \
              f"python3 asr-python/src/eval.py --model-type {model_type} --attack-dir asr-python/{attack_dir}/{filename}/adv-label-{adv_label} "

        print(cmd)
        subprocess.run(cmd, shell=True)
