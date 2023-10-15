from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from accelerate import Accelerator
from trl import DPOTrainer, SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from peft import LoraConfig, LoraModel, AutoPeftModelForCausalLM
import datetime
import wandb


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    num_train_epochs: Optional[int] = field(default=1)
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    max_grad_norm: Optional[float] = field(default=10)
    project: Optional[str] = field(default="dpo")
    run_name: Optional[str] = field(default="dpo_model")
    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    save_total_limit: Optional[int] = field(default=None)
    save_steps: Optional[float] = field(default=0.5)
    use_peft: Optional[bool] = field(default=True)
    peft_lora_r: Optional[int] = field(default=64)
    peft_lora_alpha: Optional[int] = field(default=16)
    # target_modules: Optional[list] = field(default=None)
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    # output_dir
    output_dir: Optional[str] = field(
        default="/data/avishnevskiy/experiments",
    )
    from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "checkpoint to continue from"
        }
    )


def get_experiment_name(exp_name):
    """Transform experiment name, so we have different experiments"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    exp_name = f"{exp_name}-{formatted_time}"
    return exp_name


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    exp_name = get_experiment_name(script_args.run_name)
    exp_dir = os.path.join(script_args.output_dir, exp_name)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        # initialize wandb
        run = wandb.init(
            project=script_args.project,
            name=exp_name
        )

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True
        )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Anthropic Helpful-Harmless dataset
    train_dataset = get_hh("train", sanity_check=script_args.sanity_check)

    # 3. Load evaluation dataset
    eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        evaluation_strategy="steps",
        save_total_limit=script_args.save_total_limit,
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        eval_steps=500,
        output_dir=exp_dir,
        optim="rmsprop",
        warmup_steps=150,
        report_to=script_args.report_to,
        bf16=True,
    )
    
    # 5. get lora config
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            target_modules=['c_attn', 'c_proj', 'dense_4h_to_h', 'c_fc', 'c_fc2', 'c_proj', 'lm_head'],
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # 6. initialize the DPO trainer
    # Can be different templates for different datasets
    response_template = "\n\nAssistant:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer, 
        mlm=False
    )

    sft_trainer = SFTTrainer(
        model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=script_args.max_length,
        dataset_text_field="chosen",
        peft_config=peft_config
    )

    # 6. train
    if script_args.from_checkpoint is not None:
        sft_trainer.train(script_args.from_checkpoint)
    else:
        sft_trainer.train()

    sft_trainer.save_model(os.path.join(exp_dir, 'LATEST'))

    # save LoRa model
    if script_args.use_peft:
        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        sft_trainer.model.save_pretrained(output_dir)

        # Free memory for merging weights
        del model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir, 
            trust_remote_code=True,
            device_map="auto"
        )
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)