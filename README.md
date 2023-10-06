# DPO

In order to run distributed training using accelerate and DeepSpeed:

```bash
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/dpo.py --model_name_or_path=lomahony/eleuther-pythia2.8b-hh-sft --per_device_train_batch_size=1 --gradient_accumulation_steps=16 --sanity_check=false --learning_rate=1e-6 --report_to=wandb --run_name=trl_dpo_pythia
```