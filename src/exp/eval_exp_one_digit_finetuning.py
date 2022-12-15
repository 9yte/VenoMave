import sys
import subprocess

gpu = int(sys.argv[1])
attack_dir = sys.argv[2]
model_type = sys.argv[3]
baseModelDir = sys.argv[4]

# task = 'SPEECHCOMMANDS'
task = 'TIDIGITS'

sentences = False

if task == 'TIDIGITS':

    if sentences:
        with open('src/exp/exp-sentences-pairs-len3.txt') as f:
            pairs = f.readlines()
    else:
        with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt') as f:
            pairs = f.readlines()

    data_dir = '/asr-python/data'
elif task == 'SPEECHCOMMANDS':
    with open('src/exp/speechcommands-utterances_shuffled.txt') as f:
        pairs = f.readlines()
    data_dir = '/asr-python/data_speech_commands2'
else:
    assert False

pairs = pairs[0:30]
for idx, pair in enumerate(pairs):
    filename, adv_label = pair.split()
    if True or idx % 4 == gpu:
        cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
              f"-v {attack_dir}:/results " \
              "-it sound_poisoning " \
              f"python3 asr-python/src/eval.py --data-dir {data_dir} --task {task} --victim-config cfg2-dp-0.2-finetuning-v2 " \
              f"--model-type {model_type} --attack-dir /results/{filename}/adv-label-{adv_label} " \
              f"--base-model asr-python/{baseModelDir}/model.h5 --base-hmm asr-python/{baseModelDir}/hmm.h5 "

        print(cmd)
        subprocess.run(cmd, shell=True)
