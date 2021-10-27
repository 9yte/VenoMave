#!/bin/bash
docker run --gpus device=$1 --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python \
  -w /asr-python \
	-it sound_poisoning \
	python3 src/eval_baseline.py --model-type $2
