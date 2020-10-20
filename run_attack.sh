#!/bin/bash
docker run --gpus device=$1 --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python \
	--user $(id -u):$(id -g) \
	-it sound_poisoning \
	python3 /asr-python/src/craft_poisons.py --target-filename $2 --adv-label $3
