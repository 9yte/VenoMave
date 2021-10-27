import argparse
import itertools
from pathlib import Path

digits = ['Z', 'O', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def generate_adv_pairs(data_dir, output_file):

    filenames = [f.stem for f in sorted(data_dir.glob('*.wav'))]
    import random
    random.shuffle(filenames)
    print(len(filenames))

    # Y = [torch.load(data_dir.joinpath('Y', f).with_suffix('.pt')) for f in  filenames]

    selected_files = []
    adv_labels = []
    
    with open('src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled_usenix.txt') as f:
        lines = f.readlines()
    lines = lines[:12]
    for line in lines:
        line = line.strip()
        f, a_l = line.split()
        selected_files.append(f)
        adv_labels.append(a_l)

    for _ in range(3):
        for y_original, y_target in itertools.product(digits, digits):
            if y_original == y_target:
                continue

            for f in filenames:
                if f[:-1] in [s[:-1] for s in selected_files]:
                    continue

                f_original_label = f.split("-")[-1][:-1]
                if y_original == f_original_label:
                    selected_files.append(f)
                    adv_labels.append(y_target)
                    break
            else:
                assert False

    lines = list(zip(selected_files, adv_labels))
    olds = lines[:12]
    news = lines[12:]
    random.shuffle(news)
    lines = olds + news
    with open(output_file, 'w') as f:
        f.writelines(["{} {}\n".format(f, adv_label) for f, adv_label in lines])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw/TEST/wav', type=Path)
    parser.add_argument('--output-file', default='src/exp/exp-one-digit-pairs-singledigit-utterances_shuffled.txt')

    args = parser.parse_args()

    generate_adv_pairs(args.data_dir, args.output_file)
