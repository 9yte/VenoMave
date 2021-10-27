import sys
import subprocess
from pathlib import Path

gpu = int(sys.argv[1])

task = 'SPEECHCOMMANDS'
# task = 'TIDIGITS'

if task == 'TIDIGITS':
    with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt') as f:
        pairs = f.readlines()
    data_dir = '/asr-python/data'
    net = 'TwoLayerPlus'
elif task == 'SPEECHCOMMANDS':
    with open('src/exp/speechcommands-utterances_shuffled.txt') as f:
        pairs = f.readlines()
    data_dir = '/asr-python/data_speech_commands'
    net = 'ThreeLayerPlusPlus'
else:
    assert False


pairs = pairs[0:20]
for idx, pair in enumerate(pairs):
    filename, adv_label = pair.split(" ")
    # if filename != 'TEST-MAN-NT-3B':
    #     continue
    # if filename != 'TEST-MAN-HR-9A':
    #     continue
    # if filename != 'TEST-MAN-NP-9A':
    #     continue
    if idx % 4 == gpu:
        cmd = "docker run --gpus device={} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
              "-it sound_poisoning " \
              "python3 -W ignore /asr-python/src/craft_poisons.py " \
              "--target-filename {} --adv-label {} --data-dir {} --task {} --poisons-budget {} --model-type {} ".format(gpu, filename, adv_label.strip(), data_dir, task, 0.02, net)

        print(cmd)
        subprocess.run(cmd, shell=True)
