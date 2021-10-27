import argparse
import itertools
from pathlib import Path

# words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'backward', 'bed', 'bird',
# 'cat', 'dog', 'down', 'follow', 'forward', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'no', 'off', 'on',
# 'right', 'sheila', 'stop', 'tree', 'up', 'visual', 'wow', 'yes']
words = ['zero', 'one', 'two', 'three', 'four',
         'five', 'six', 'seven', 'eight', 'nine', 'backward', 'down', 'follow', 'forward',
         'go', 'learn', 'left', 'no', 'off', 'on', 'right', 'stop', 'up',
         'yes']


def generate_adv_pairs(data_dir, output_file):
    filenames = [f.stem for f in sorted(data_dir.glob('*.wav'))]
    import random
    random.shuffle(filenames)
    print(len(filenames))

    # Y = [torch.load(data_dir.joinpath('Y', f).with_suffix('.pt')) for f in  filenames]

    selected_files = []
    adv_labels = []

    for y_original, y_target in itertools.product(words, words):
        if y_original == y_target:
            continue

        for f in filenames:
            if f in [s for s in selected_files]:
                continue

            f_original_label = f.split("_")[0]
            if y_original == f_original_label:
                selected_files.append(f)
                adv_labels.append(y_target)
                break
        else:
            assert False

    lines = list(zip(selected_files, adv_labels))
    random.shuffle(lines)
    with open(output_file, 'w') as f:
        f.writelines(["{} {}\n".format(f, adv_label) for f, adv_label in lines])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_speech_commands/raw/TEST/wav', type=Path)
    parser.add_argument('--output-file', default='src/exp/speechcommands-utterances_shuffled.txt')

    args = parser.parse_args()

    generate_adv_pairs(args.data_dir, args.output_file)
