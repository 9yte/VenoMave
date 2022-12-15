#!/bin/bash
docker run --gpus device=$1 --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python \
	-v /data/hojjat/sound-poisoning/results-paper:/asr-python-2 \
	-it sound_poisoning \
	python3 /asr-python/src/print_SNRseg.py --data-dir /asr-python/data --poison-dir /asr-python/$2 --model-type $3
