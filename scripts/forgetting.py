from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import pickle
import numpy as np
import gc


MAX_SIZE = 8000


def move_to_gpu(batch):
    return {k: v.cuda() for k, v in batch.items()}

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_SIZE)

def main(model, name):
    attentions, logits, loss = [], [], []

    for batch in tqdm(data_loader):
        with torch.no_grad():
            output = model(**move_to_gpu(batch))
            loss.append(output.loss.cpu().numpy())

    concatenated = np.concatenate([np.array([arr.item()]) for arr in loss])
    np.save(f'{name}.npy', concatenated)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('cerebras/btlm-3b-8k-base')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('DKYoon/SlimPajama-6B', split='test', cache_dir='/data/avishnevskiy/.cache/hf')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size = 1, shuffle=False)
    
    loss = torch.nn.CrossEntropyLoss()

    model_list = [
        ('/data/avishnevskiy/experiments/dpo_btlm-20231017-054122/LATEST/', 'btlm_0.1'),
        ('/data/avishnevskiy/experiments/dpo_btlm_shp-20231026-214852/LATEST/', 'btlm_shp_0.1'),
        ('/data/avishnevskiy/experiments/dpo_btlm-20231024-232611/LATEST/', 'btlm_0.3'),
        ('/data/avishnevskiy/experiments/dpo_btlm-20231025-163346/LATEST/', 'btlm_0.5'),
        ('/data/avishnevskiy/experiments/dpo_btlm-20231027-083432/LATEST/', 'btlm_shp_0.3')
    ]

    for model_path, name in model_list:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
        model.eval()
        main(model, name)

        del model
        gc.collect()
        torch.cuda.empty_cache()