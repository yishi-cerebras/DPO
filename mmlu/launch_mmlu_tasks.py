PYTHON="python"
CKPT_DIR = "/data/avishnevskiy/experiments/trl_dpo_pythia_fp16-20231011-010059"
checkpoints = [
    "checkpoint-500",
    "checkpoint-1000",
    "checkpoint-1509",
    "checkpoint-2012",
    "LATEST"
]

import os
checkpoints = [
        os.path.join(CKPT_DIR, ckpt_file) for ckpt_file in checkpoints
]

for ckpt in checkpoints:
    output_file = f"pythia28_{ckpt.split('/')[-1]}_mmlu_5_shot.out"
    cmd = f"{PYTHON} run_mmlu.py --checkpoint_dir {ckpt} --ntrain 5 --output_path {output_file}"
    os.system(cmd)
