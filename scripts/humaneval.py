import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords:list, start_length:int, batch_size:int, tokenizer):
        self.keywords = keywords
        self.state = [False]*batch_size
        self.start_length = start_length
        self.end_token = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for i in range(input_ids.shape[0]):
            if self.state[i]: continue
            if input_ids[i][-1] == self.end_token:
                self.state[i] = True
            else:
                sample = self.tokenizer.decode(input_ids[i][self.start_length:])
                self.state[i] = any([keyword in sample for keyword in self.keywords])
        return all(self.state)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
        device_map=torch.device(args.device),
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )

    stop_words = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
    dataset = load_dataset("openai_humaneval", split="test")
    fp = open(args.target_path, "w")

    for task in tqdm(dataset):
        if int(task["task_id"].split("/")[-1]) < args.resume: continue
        prompt = task["prompt"].replace("    ", "\t") if args.format_tabs else task["prompt"]
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False,).to(args.device)
        for i in range(8):
            batch_size = 25
            stop_criteria = KeywordsStoppingCriteria(stop_words, inputs.input_ids.shape[1], batch_size, tokenizer)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=batch_size,
                do_sample=True,
                top_p=0.95,
                temperature=args.temp,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            for out in outputs:
                sample = tokenizer.decode(out.tolist()[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                record = {"task_id": task["task_id"], "completion": sample}
                fp.write(json.dumps(record) + "\n")
    fp.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained")
    parser.add_argument("target_path")
    parser.add_argument("device")
    parser.add_argument("temp", type=float)
    parser.add_argument("resume", type=int)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--format_tabs", action="store_true")
    args = parser.parse_args()
    main(args)

