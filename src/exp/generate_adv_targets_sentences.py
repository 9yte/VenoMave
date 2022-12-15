import random
import argparse
import itertools
from pathlib import Path

digits = ['Z', 'O', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def random_replace(digit, blocklist):
    assert digit in digits

    while True:
        random_digit = random.sample(digits, k=1)[0]

        if random_digit != digit and random_digit not in blocklist:
            return random_digit


def generate_adv_pairs(data_dir, seed, output_file):
    random.seed(seed)

    filenames = [f.stem for f in sorted(data_dir.glob('*.lab'))]
    random.shuffle(filenames)

    print(len(filenames))

    # Y = [torch.load(data_dir.joinpath('Y', f).with_suffix('.pt')) for f in  filenames]

    selected_files = []
    adv_labels = []

    for f in filenames:

        f_original_label = f.split("-")[-1][:-1]

        if len(set(list(f_original_label))) != len(f_original_label):
            continue

        if len(f_original_label) == 4:

            f_adv_label = ''.join([random_replace(y_original, f_original_label) for y_original in f_original_label])
            selected_files.append(f)
            adv_labels.append(f_adv_label)

    with open(output_file, 'w') as f:
        f.writelines(["{} {}\n".format(f, adv_label) for f, adv_label in zip(selected_files, adv_labels)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=171717)
    parser.add_argument('--data-dir', default='data/raw/TEST/lab', type=Path)
    parser.add_argument('--output-file', default='src/exp/exp-sentences-pairs.txt')

    args = parser.parse_args()

    generate_adv_pairs(args.data_dir, args.seed, args.output_file)
