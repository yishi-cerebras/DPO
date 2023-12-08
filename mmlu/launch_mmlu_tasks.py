PYTHON="python"
# CKPT_DIR = "/data/avishnevskiy/experiments/dpo_btlm_alpaca-20231202-062603"
# checkpoints = [
#     "LATEST"
# ]

import os
# checkpoints = [
#         os.path.join(CKPT_DIR, ckpt_file) for ckpt_file in checkpoints
# ]

checkpoints = [
    # ('/data/avishnevskiy/experiments/sft-btlm', 'sft-btlm'),
    # ('EleutherAI/pythia-2.8b', 'pythia2.8'),
    # ('lomahony/eleuther-pythia2.8b-hh-sft', 'pythia2.8-sft'),
    ('/data/avishnevskiy/experiments/trl_dpo_pythia-20231130-013042/LATEST', 'pythia2.8-dpo')
]

for ckpt, name in checkpoints:
    output_file = f"{name}_mmlu_5_shot.out"
    cmd = f"{PYTHON} run_mmlu.py --checkpoint_dir {ckpt} --ntrain 5 --output_path {output_file}"
    os.system(cmd)
