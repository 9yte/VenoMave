import argparse
from pathlib import Path
import json
import re

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--expjson-dir', default='exp/order_test.json', type=Path)

    params = parser.parse_args()


    exp_base_dir = Path("_adversarial_usenix").joinpath(params.expjson_dir.stem)

    exp_dict = json.loads(params.expjson_dir.read_text())
    
    print("-"*40)

    for time_stamp_dir in exp_base_dir.glob("*"):
        print(f"[+] time stamp {time_stamp_dir.stem}")

        for experiment_name, experiment_param in exp_dict.items():
            
            exp_dir = time_stamp_dir.joinpath(experiment_name)

            if exp_dir.exists():
                print(f"[+] results for experiment {experiment_name}")

                for log_dir in exp_dir.rglob("log.txt"):
                    log_text = log_dir.read_text()

                    pattern = re.compile(r"Early stopping of the attack after [0-9]+ steps")

                    for entry in re.finditer(pattern, log_text):
                        iter_suc = entry.group(0).split()[-2]
                        print(f" -> Poisoning successful after {iter_suc} iterations!")
                        print()
                
                for snr_dir in exp_dir.rglob(f"{iter_suc}/snrseg.json"):
                    snrseg_dict = json.loads(snr_dir.read_text())

                    snrseg_list = []
                    for wav_file, snrseg in snrseg_dict.items():
                        snrseg_list += [np.float(snrseg)]
                    
                    print(f" -> mean snrseg:    {np.mean(snrseg_list)}")
                    print(f" -> min snrseg: {np.min(snrseg_list)}")
                    print(f" -> max snrseg: {np.max(snrseg_list)}")

            print()
        
        print("-"*40)
            
