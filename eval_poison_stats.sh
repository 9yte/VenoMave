#!/bin/bash
docker run --gpus device=$1 --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python \
	-v /data/hojjat/sound-poisoning/results-paper:/asr-python-2 \
	--user $(id -u):$(id -g) \
	-it sound_poisoning \
	python3 /asr-python/src/eval_stats.py --data-dir /asr-python/data --exp-dir /asr-python/$2 --model-type $3
