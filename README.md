# VENOMAVE
This repository provides datasets and codes that are needed to reproduce the experiments in the paper [VENOMAVE: Targeted Poisoning Against Speech Recognition](https://arxiv.org/pdf/2010.10682.pdf).

## Overview
VENOMAVE is the first data poisoning attack in the audio domain. The wide adoption of Automatic Speech Recognition (ASR) remarkably enhanced human-machine interaction. Prior research has demonstrated that modern ASR systems are susceptible to adversarial examples, i.e., malicious audio inputs that lead to misclassification by the victim's model at run time. The research question of whether ASR systems are also vulnerable to data-poisoning attacks is still unanswered. In such an attack, a manipulation happens during the training phase: an adversary injects malicious inputs into the training set to compromise the neural network's integrity and performance. Prior work in the image domain demonstrated several types of data-poisoning attacks, but these results cannot directly be applied to the audio domain. In this paper, we present the first data-poisoning attack against ASR, called VenoMave. We evaluate our attack on an ASR system that detects sequences of digits. When poisoning only 0.17% of the dataset on average, we achieve an attack success rate of 86.67%. To demonstrate the practical feasibility of our attack, we also evaluate if the target audio waveform can be played over the air via simulated room transmissions. In this more realistic threat model, VenoMave still maintains a success rate up to 73.33%. We further extend our evaluation to the Speech Commands corpus and demonstrate the scalability of VenoMave to a larger vocabulary. During a transcription test with human listeners, we verify that more than 85% of the original text of poisons can be correctly transcribed. We conclude that data-poisoning attacks against ASR represent a real threat, and we are able to perform poisoning for arbitrary target input files while the crafted poison samples remain inconspicuous.

If you find this code useful for your research you may cite our paper
```
@article{aghakhani2020venomave,
  title={VENOMAVE: Clean-Label Poisoning Against Speech Recognition},
  author={Aghakhani, Hojjat and Eisenhofer, Thorsten and Sch{\"o}nherr, Lea and Kolossa, Dorothea and Holz, Thorsten and Kruegel, Christopher and Vigna, Giovanni},
  journal={arXiv preprint arXiv:2010.10682},
  year={2020}
}
```

## Poison examples
As an example, we have put a few poison samples for your review. Download the samples from [here](https://drive.google.com/file/d/1DSzzPXo5tn3mSCozxF-zRlpzC0ZfWgj_/view?usp=sharing) and unzip the file.
In the folder `samples/poisons`, you can find 20 randomly selected poison samples for 12 succesfull attacks. We provided poison samples for both when the psychoacoustic modeling is disabled as well as when a margin of 30dB is used.
In the foler `samples/clean`, you can find the original (i.e., clean) poison samples that are in the `samples/poisons` folder.

## Prerequisites
### Dataset
#### TIDIGTS
Before anything, download our datasets and surrogate decoders (HMMs) from [here](https://drive.google.com/file/d/1H7akxsdFISEzbRMmMGu4bmJWWsl3Ahyw/view?usp=sharing). Unzip it in the root directory.
Then, the raw audio files of the dataset can be found at `data/raw`. In the example that we are attacking the victim's system with `DNN2+` network, the surrogate decoder/HMM is located at `data/TwoLayerPlus/aligned/hmm.h5`. The aligned labels (alignment for audio waveform and HMM states) for training and test sets are located at `data/TwoLayerPlus/aligned/TRAIN` and `data/TwoLayerPlus/aligned/TEST` directories. It should be noted that the `TEST` directory will not be used except for the targeted input file. That is, when evaluating the victim's system, we do not use these aligned labels since we train the whole system from scratch. This guarantees the fairness of evaluation, with respect to our threat model.

If you want, you can download the TIDIGITS dataset yourself from [here](https://catalog.ldc.upenn.edu/LDC93S10), just note that it does not come with the alignment of labels. We have used the `Montreal Forced Aligner` library to determine which parts of audio waveform correspond to which digit (in the ground-truth label). You can download the specific version of the library that we've used to generate the alignments for the dataset [here](https://drive.google.com/file/d/1J-mtUf9l0ySFEatLO-6LCiYWzt4klfcE/view?usp=sharing). Then unzip it into folder `montreal-forced-aligner`.

If you download the dataset from our link, you don't need to download and run this library yourself.

#### SPEECH COMMANDS
Download our datasets and surrogate decoders (HMMs) from [here](https://drive.google.com/file/d/1nIzGFuZvLcMTVQIu7-c8FcNOtqD1NQ__/view?usp=sharing). Unzip it in the root directory.

### Experiments
To run experiments in the docker, you first need to build the docker image.
```
docker build -t sound_poisoning .
```

Note that the default docker file (i.e., `Dockerfile`) is using `Cuda 10.1`. We also provide `Dockerfile-cuda11.0`, which uses `Cuda 11.0`

You can start the poisoning attack by running:
```
docker run --gpus device=$gpu --rm -v $REPO_PATH:/asr-python \
	-it sound_poisoning \
	"--target-filename $targetFilename --adv-label $advLabel --data-dir $dataDir --task $task --poisons-budget $poisonBudget --model-type $net "
```
where `$dataDir` is the path to the data directory, and `$task` is either `TIDIGITS` or `SPEECHCOMMANDS`. The parameter `$net` determines the network type we use for the surrogate networks, and `$poisonBudget` is the poison budget we use (r_p in the paper). Parameters `$targetFilename` and `$advLabel` determine the name of the targeted input file and the adversarial word sequence, respectively.
You may want to look at the list of parameters of `src/craft_poisons.py`.

You can use `eval.sh` script to evaluate the generated poisons of one attack against a victim that starts training the ASR system from scratch. Even the Viterbi training is being done from scratch. The random seed used by the victim is new.

To automate the experiments in our paper, we have used scripts in `src/exp/`, which you may find useful. You probably need to adjust the path to the source code.

### Results
We have uploaded all individual attack examples that are reported in the paper (including poison examples and log files) [here](https://drive.google.com/file/d/1Frs9PG40oNwdTDCuz7BZLGi3z5NiwVc1/view?usp=sharing).

##  If you have questions, feel free to reach us.
