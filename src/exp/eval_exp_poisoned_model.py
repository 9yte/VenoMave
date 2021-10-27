import sys
import subprocess

gpu = int(sys.argv[1])
attack_dir = sys.argv[2]
model_type = sys.argv[3]

cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
      "--user $(id -u):$(id -g) " \
      "-it sound_poisoning " \
      f"python3 asr-python/src/eval_poisoned_model.py --victim-net {model_type} --attack-dir asr-python/{attack_dir} "

print(cmd)
subprocess.run(cmd, shell=True)
