import sys
import subprocess

gpu = int(sys.argv[1])
attack_dir = sys.argv[2]
model_type = sys.argv[3]

task = 'SPEECHCOMMANDS'
# task = 'TIDIGITS'

if task == 'TIDIGITS':
    with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt') as f:
        pairs = f.readlines()
    data_dir = '/asr-python/data'
elif task == 'SPEECHCOMMANDS':
    with open('src/exp/speechcommands-utterances_shuffled.txt') as f:
        pairs = f.readlines()
    data_dir = '/asr-python/data_speech_commands2'
else:
    assert False

pairs = pairs[0:10]
for idx, pair in enumerate(pairs):
    filename, adv_label = pair.split()
    if idx % 4 == gpu:
        cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
              "-v /data2/hojjat/sound-poisoning:/asr-python-2 " \
              "-it sound_poisoning " \
              f"python3 asr-python/src/eval.py --data-dir {data_dir} --task {task} --victim-config cfg2-dp-0.2 " \
              f"--model-type {model_type} --attack-dir asr-python/{attack_dir}/{filename}/adv-label-{adv_label} "

        print(cmd)
        subprocess.run(cmd, shell=True)
