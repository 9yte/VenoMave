import random
import argparse
import itertools
from pathlib import Path

digits = ['Z', 'O', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def generate_adv_pairs(data_dir, adv_pair_repeat_num, seed, output_file):
    random.seed(seed)

    filenames = [f.stem for f in sorted(data_dir.glob('*.pt'))]
    random.shuffle(filenames)

    print(len(filenames))

    # Y = [torch.load(data_dir.joinpath('Y', f).with_suffix('.pt')) for f in  filenames]

    selected_files = []
    adv_labels = []
    for _ in range(adv_pair_repeat_num):
        for y_original, y_target in itertools.product(digits, digits):
            if y_original == y_target:
                continue

            for f in filenames:
                if f in selected_files:
                    continue

                f_original_label = f.split("-")[-1][:-1]
                if y_original in f_original_label:
                    f_adv_label = f_original_label.replace(y_original, y_target, 1)
                    selected_files.append(f)
                    adv_labels.append(f_adv_label)
                    break

    with open(output_file, 'w') as f:
        f.writelines(["{} {}\n".format(f, adv_label) for f, adv_label in zip(selected_files, adv_labels)])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=171717)
    parser.add_argument('--data-dir', default='data/aligned/TEST/Y', type=Path)
    parser.add_argument('--adv-pair-repeat-num', default=3, type=int)
    parser.add_argument('--output-file', default='src/exp/exp-one-digit-pairs.txt')

    args = parser.parse_args()

    generate_adv_pairs(args.data_dir, args.adv_pair_repeat_num, args.seed, args.output_file)