from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, pipeline
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length
from tqdm import tqdm
import torch
import json
import sys
sys.path.append('..')
from dpo import get_hh, get_shp
import argparse
import json
import types
# from fuzzywuzzy import fuzz
from datasets import load_dataset
from collections import defaultdict
import random


MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256

def get_model(checkpoint):
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    return model

def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_dataset(dataset_name):
    if dataset_name == 'shp':
        eval_dataset = get_shp("test", sanity_check=False)
    else:
        eval_dataset = get_hh("test", sanity_check=False)
    return eval_dataset

def get_batch_samples(self, model, tokenizer, batch, temperature = 1):
    """Generate samples from the model and reference model for the given batch of inputs."""

    policy_output = model.generate(
        batch["prompt_input_ids"],
        attention_mask=batch["prompt_attention_mask"],
        max_length=MAX_LENGTH,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature
    )

    policy_output = pad_to_length(policy_output, MAX_LENGTH, tokenizer.pad_token_id)
    policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
    return policy_output_decoded

def get_trainer(model, eval_dataset, tokenizer):
    training_args = TrainingArguments(
        do_train=False,
        do_predict=True,
        remove_unused_columns=False,
        save_strategy="steps",
        do_eval=True,
        save_steps=0.2,
        output_dir='.',
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        save_total_limit=2,
        report_to=None,
    )
    trainer = DPOTrainer(
        model,
        model,
        args = training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    trainer.get_batch_samples = types.MethodType( get_batch_samples, trainer )
    return trainer

def get_categories_index():
    categories_indxs = []
    with open('categories_indx.txt', 'r') as f:
        for line in f:
            categories_indxs.append(int(line))
    return categories_indxs

def get_generations(model, eval_dataset, tokenizer, dataset_name, only_last=False):
    generations = []
    categories_indxs = get_categories_index()
    trainer = get_trainer(model, eval_dataset, tokenizer)
    eval_loader = trainer.get_eval_dataloader()

    for i, b in tqdm(enumerate(eval_loader)):
        if dataset_name == 'shp':
            if i not in categories_indxs:
                continue
        elif dataset_name == 'hh':
            if only_last and i < len(eval_loader)-256:
                continue
            if i >= 256 and i < len(eval_loader)-256:
                continue
        else:
            raise NotImplementedError
        
        policy = trainer.get_batch_samples(model, tokenizer, b, 1)
        assistant_word = '\n\nAssistant:'
        resp_indx = policy[0].rfind(assistant_word) # TODO: change for another one, policy can output without Assistant word
        prompt = b['prompt'][0][:resp_indx]
        policy_response = policy[0][resp_indx+len(assistant_word):].strip()
        chosen_response = b['chosen'][0][resp_indx+len(assistant_word):].strip()
        generations.append({'prompt': prompt, 'chosen_response': chosen_response, 'policy_response': policy_response})
    return generations

def main(checkpoint, dataset_name, tokenizer_name, generation_name, only_last):
    model = get_model(checkpoint)
    eval_dataset = get_dataset(dataset_name)
    tokenizer = get_tokenizer(tokenizer_name)
    generations = get_generations(model, eval_dataset, tokenizer, dataset_name, only_last)

    with open(f'../generations/{generation_name}-temp1.json', 'w') as json_file:
        json.dump(generations, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--tokenizer_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--generation_name', type=str)
    parser.add_argument('--only_last', type=bool, default=False)
    args = parser.parse_args()
    main(args.checkpoint, args.dataset_name, args.tokenizer_name, args.generation_name, args.only_last)