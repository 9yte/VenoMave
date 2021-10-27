import sys
import subprocess

gpu = int(sys.argv[1])
attack_dir = '_adversarial_paper_usenix2022/TIDIGITS/speakers-0-55/ratioAnalysis/8-sub-models/TwoLayerPlus/budget-0.005/no-psycho-offset'
model_type = 'ThreeLayer'

cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
      "-v /data2/hojjat/sound-poisoning:/asr-python-2 " \
      "-it sound_poisoning " \
      f"python3 asr-python/src/eval_overtheair.py --victim-net {model_type} --attack-root-dir asr-python-2/{attack_dir} "

print(cmd)
subprocess.run(cmd, shell=True)
