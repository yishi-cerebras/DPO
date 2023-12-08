# DPO

This repository provides code and instructions for training a DPO model. While it includes code for SFT training, I recommend using the SFT code from the original [repository](https://github.com/eric-mitchell/direct-preference-optimization). There have been some issues with the SFT code provided here.


### Project layout

**mmlu** - directory containing code for running MMLU task.

**longeval** - directory contating code for running [LongEval](https://github.com/DachengLi1/LongChat).

**notebooks** - notebooks used for generating scripts, graphs and etc. Currently not used.


**scripts** - all the scripts needed for:
1. DPO, SFT training
2. Win-rate evaluation
3. Forgetting evaluation using SlimPajama dataset
4. HumanEval
5. Helper scripts

### How to run DPO training

You have the option to run DPO training with or without LoRa. By default, LoRa is disabled. To enable the LoRa model, you need to set the flag `--use_peft=true`. If you wish to modify LoRa parameters, you should also provide values for `--peft_lora_r` and `--peft_lora_alpha`. Additionally, you may find it helpful to review all parameters in `scripts/dpo.py` for a comprehensive understanding.

At present, distributed training is exclusively supported through Deepspeed. There are two configurations available for Deepspeed. Stage 3 is more memory-efficient, although in some cases, Stage 2 may yield better results. 

In order to run distributed training:

```bash
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info CUDA_VISIBLE_DEVICES=<list all gpus> accelerate launch --config_file=configs/deepspeed_zero{2,3}.yaml scripts/dpo.py --model_name_or_path=<model_path> 
--tokenizer_name=<tokenizer_name>
--dataset_name=<dataset_name>
--beta=<beta>
--per_device_train_batch_size=<device_train_batch_size> --gradient_accumulation_steps=<gradient_accumulation_steps> --sanity_check=false --learning_rate=<learning_rate> --num_train_epochs=<num_epochs> --report_to=wandb --run_name=<run_name>
```

In order to run job on single GPU:

```bash
python scripts/dpo.py --model_name_or_path=<model_path>  --per_device_train_batch_size=<device_train_batch_size>  --gradient_accumulation_steps=<gradient_accumulation_steps> --dataset_name=<dataset_name> --sanity_check=false --learning_rate=<learning_rate> --peft_lora_r=128 --report_to=wandb --run_name=<run_name> --beta=<beta> --num_train_epochs=<num_epochs> --tokenizer_name=<tokenizer_name>
```

### How to run SFT training

The same way you run DPO training but instead of scripts/dpo.py you should you scripts/sft.py

### How to run GPT-4 evaluation

1. To run evaluations, you have two options: `evaluation.py` for a single model or `evaluation_multiple.py` for multiple models. For either, you'll need to specify the checkpoint, tokenizer name, and dataset name (as defined in dpo.py), along with a name for the run. The last argument is specific to the Antropic-HH dataset, allowing you to evaluate the last 256 samples. If this argument is omitted, the evaluation will default to the first 256 and last 256 examples.

2. To proceed, you must obtain an `OpenAI key` and set it as the environment variable `OPENAI_KEY`. Then, run `chatgpt-eval.py`, using the run name specified in the previous step.

### Model weights:

All model weights are stored on CoreWeave in the /data/avishnevskiy/experiments/[name of the experiment]. You can find the specific name of the experiment on the [WandB page](https://wandb.ai/alexander-vishnevskiy/dpo).


### Additional libraries used:

For Eleuther harness evaluation: https://github.com/EleutherAI/lm-evaluation-harness

direct-preference-optimixation: https://github.com/eric-mitchell/direct-preference-optimization

FastChat: https://github.com/lm-sys/FastChat for MT-Bench score