import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM

from trl import DPOTrainer
from accelerate import Accelerator
# import wandb


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
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    lora_r: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)
    lora_alpha: Optional[int] = field(default=16)
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

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

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)

if __name__ == "__main__":
    # Start a W&B Run
    # run = wandb.init(
    #     project="dpo",
    #     name="pythia28_dpo_trl"
    # )
    
    script_args = ScriptArguments(
        model_name_or_path = "lomahony/eleuther-pythia2.8b-hh-sft",
        sanity_check = False,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 64,
        learning_rate = 1e-6,
        # report_to = "wandb"
    )
    
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, 
        torch_dtype=torch.float16, 
        device_map = {"": Accelerator().local_process_index}
    )
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
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        fp16=True,
        remove_unused_columns=False,
        save_strategy="epoch",
        run_name="pythia28_dpo_trl",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        num_train_epochs = 1,
        logging_steps=10,  # match results in blog post
        eval_steps=20000,
        output_dir="./test",
        optim="adamw_torch",
        warmup_steps=150,
        report_to=script_args.report_to,
    )
    
    # 5. Peft config
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 6. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        peft_config=peft_config
    )
    
    # 6. train
    dpo_trainer.train()