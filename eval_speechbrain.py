import os
import sys
import subprocess
from pathlib import Path

res_root_path = Path(sys.argv[1])
dataset = 'speechcommands'
yaml_file = 'xvect.yaml'


GPUS = ["0", "1", "2", "3"]

for idx, poisons_json_p in enumerate(res_root_path.glob("**/poisons.json")):

    gpu = int(sys.argv[2])
    
    if idx % 4 == gpu:

        gpu = GPUS[gpu]

        d = poisons_json_p.parent
        all_poison_step_dirs = [s for s in d.iterdir() if s.is_dir()]

        last_step = max([int(a.name) for a in all_poison_step_dirs])

        poisons_dir = d / str(last_step)

        output_dir = f'{poisons_dir}/speechbrain-model'
        if os.path.exists(output_dir + "/log.txt"):
            continue

        cmd = f'python src/speechbrain/{dataset}/prepare_dataset.py --clean-data-dir data/raw/ --poison-data-dir {poisons_dir}'

        print(cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'python src/speechbrain/{dataset}/train.py src/speechbrain/{dataset}/{yaml_file} --device=cuda --gpu {gpu} --train_annotation={poisons_dir}/dataset-prepared-for-speechbrain-train.json --test_annotation={poisons_dir}/dataset-prepared-for-speechbrain-test.json --target_filename {poisons_dir.parent.parent.parent.name} --output_folder {output_dir}'
     
        print(cmd)
        subprocess.run(cmd, shell=True)

