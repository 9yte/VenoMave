import sys
import subprocess

gpu = int(sys.argv[1])
attack_dir = sys.argv[2]
model_type = sys.argv[3]

cmd = f"docker run --gpus device={gpu} --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python " \
      f"-v {attack_dir}:/attack_dir " \
      "-it sound_poisoning " \
      f"python3 asr-python/src/eval_overtheair_realroom.py --data-dir /asr-python/data_speech_commands --victim-net {model_type} --attack-root-dir /attack_dir "

print(cmd)
subprocess.run(cmd, shell=True)
