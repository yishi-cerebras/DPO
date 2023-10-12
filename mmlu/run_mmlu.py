import argparse
import json
import os
import time

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding=True
    )
    input_tokens = {
        k: input_tokens[k]
        for k in input_tokens
        if k in ["input_ids", "attention_mask"]
    }
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda")

    return input_tokens


def load(ckpt_dir):
    n_gpus = torch.cuda.device_count()

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir, padding_side="left"
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1

    # get model, requires accelerate for using low cpu memory
    #model = AutoModelForCausalLM.from_pretrained(
    #        ckpt_dir, torch_dtype="auto", low_cpu_mem_usage=True, trust_remote_code=True,device_map="cuda:0",
    #)
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage=True, trust_remote_code=True,device_map="cuda:0") 
    model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    model.eval()

    return model, tokenizer


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 2
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(
            tokenizer.batch_decode(outputs, skip_special_tokens=True)
        )
    answers = [answer[-1] for answer in answers]
    return answers


def main(data_dir: str, ckpt_dir: str, output_filename: str, ntrain: int = 5):
    run_results = {}

    model, tokenizer = load(ckpt_dir)
    start_time = time.time()
    for task in TASKS:
        print(f"Testing {task} ...")
        records = []
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", task + "_dev.csv"), header=None
        )[:ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", task + "_test.csv"), header=None
        )
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})

        pred_answers = batch_infer(
            model, tokenizer, [record["prompt"] for record in records]
        )
        gold_answers = [record["answer"] for record in records]
        run_results[task] = {
            "pred_answers": pred_answers,
            "gold_answers": gold_answers,
        }

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    end_time = time.time() - start_time
    print(f"Total run time: {end_time}")


def get_arguments():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--data_dir", type=str, default="/data/daria/mmlu_data/",
        help="Data provided in data/ folder, preferably do not change")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
        help="Location of huggingface checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
        help="Location of output file used to compute metrics, in .json format")
    parser.add_argument("--ntrain", type=int, default=5,
        help="Number of few_shot to use for eval, defaults to 5")
    # fmt: on
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args.data_dir, args.checkpoint_dir, args.output_path, args.ntrain)
