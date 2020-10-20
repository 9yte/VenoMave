# VENOMAVE
This repository provides datasets and codes that are needed to reproduce the experiments in the paper [VENOMAVE: Clean-Label Poisoning Against Speech Recognition](Here).

VENOMAVE is the first data poisoning attack in the audio domain. Prior work in the image domain demonstrated several types of data poisoning attacks, but they cannot be applied to the audio domain. The main challenge is that we need to attack a time series of inputs. To enforce a targeted misclassification in an ASR system, we need to carefully generate a specific sequence of disturbed inputs for the target utterance, which will eventually be decoded to the desired sequence of words. More specifically, the adversarial goal is to produce a series of misclassification tasks and in each of them, we need to poison the system to misrecognize each frame of the target file. To demonstrate the practical feasibility of our attack, we evaluate VENOMAVE on an ASR system that detects sequences of digits from 0 to 9. When poisoning only 0.94% of the dataset on average, we achieve an attack success rate of 83.33%. We conclude that data poisoning attacks against ASR systems represent a real threat that needs to be considered.

If you find this code useful for your research you may cite our paper
```
```

## Poison examples
As an example, we have put a few poison samples [here](https://drive.google.com/file/d/18COxPPrjoAg-VV1m-DqeVjPUo-W-c0nk/view?usp=sharing). In this example, VENOMAVE crafts poison samples to fool the victim's system to misrecognize the sample `TEST-MAN-HR-9A` as digit `1`. In this case, the attack transfers to 70.59% of the utterances spoken by person `MAN-HR` containing digit `9`.

## Prerequisites
Before anything, download our datasets and surrogate decoders (HMMs) from [here](https://drive.google.com/file/d/1_Gog5NKwfdot3fBPyxshe9igvwzRQWsI/view?usp=sharing). Then unzip it in the root directory.
Then, the raw audio files of the dataset can be found at `data/raw`. In the example we are attacking the victim's system with `DNN2+` network, the surrogate decoder/HMM is located at `data/TwoLayerPlus/aligned/hmm.h5`. The aligned labels for training and test sets are located at `data/TwoLayerPlus/aligned/TRAIN` and `data/TwoLayerPlus/aligned/TEST` directories. It should be noted that the `TEST` directory will not be used except for the targeted input file. That is, when evaluating the victim's system, we do not use these aligned labels since we train the whole system from scratch. This guarantees the fairness of evaluation, with respect to our threat model.

If you want, you can download the TIDIGITS dataset yourself from [here](https://catalog.ldc.upenn.edu/LDC93S10), just note that it does not come with the alignment of labels. We have used the `Montreal Forced Aligner` library for that. You can download the specific version of the library that we've used to generate the alignments for the dataset [here](https://drive.google.com/file/d/1J-mtUf9l0ySFEatLO-6LCiYWzt4klfcE/view?usp=sharing). Then unzip it into folder `montreal-forced-aligner`.

If you download the dataset from our link, you don't need to download and run this library yourself.

To run experiments in the docker, you first need to build the docker image.
```
docker build -t sound_poisoning .
```

You can start the poisoning attack by running:
``` 
docker run --gpus device=$1 --rm -v /home/hojjat/audio-poison/sound-poisoning:/asr-python \
	--user $(id -u):$(id -g) \
	-it sound_poisoning \
	python3 /asr-python/src/craft_poisons.py --target-filename $2 --adv-label $3
```
where `$1`, `$2`, and `$3` are the gpu device id, name of the targeted input file, and the adversarial word sequence.
You may want to look at the list of parameters of `src/craft_poisons.py`.

You can use `eval.sh` script to evaluate the generated poisons of one attack against a victim that starts training the ASR system from scratch. Even the Viterbi training is being done from scratch. The random seed used by the victim is new.

To automate the experiments in our paper, we have used scripts in `src/exp/`, which you may found useful.

##  If you have questions, feel free to reach us.
