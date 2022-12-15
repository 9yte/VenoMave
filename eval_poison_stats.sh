#!/bin/bash
docker run --gpus device=$1 --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python \
	-v $2:/asr-python/results \
	-it sound_poisoning \
	python3 /asr-python/src/eval_poison_stats.py --data-dir /asr-python/data --exp-dir /asr-python/results --model-type $3
