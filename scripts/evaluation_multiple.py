from evaluation import main
import os

checkpoints = [
    # '/data/avishnevskiy/experiments/pythia410-sft',
    '/data/avishnevskiy/experiments/dpo_btlm_alpaca-20231202-062603/LATEST
]

dataset_names = [
    # 'hh',
    'hh',
    'hh'
    # 'hh',
    # 'hh'
]

tokenizer_names = [
    # 'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'EleutherAI/pythia-410m'
    # 'EleutherAI/pythia-2.8b',
    # 'EleutherAI/pythia-2.8b',
    # 'EleutherAI/pythia-2.8b'
]

generation_names = [
    # 'dpo_pythia410-checkpoint0',
    'dpo_pythia1.4-checkpoint0',
    'dpo_pythia2.8-checkpoint0'
    # 'dpo_btlm_sft_alpaca_on_shp_all_categories',
    # 'sft_alpaca_on_hh',
    # 'dpo_pythia410-checkpoint1800_on_hh'
    # 'dpo_pythia2.8-checkpoint2500',
    # 'dpo_pythia2.8-checkpoint3200',
    # 'dpo_pythia410-checkpoint5000',
]

only_lasts = [
    # False,
    False,
    False,
    # False
]

if __name__ == "__main__":
    for checkpoint, dataset_name, tokenizer_name, generation_name, only_last in zip(checkpoints, dataset_names, tokenizer_names, generation_names, only_lasts):
        print(f'generating for checkpoint: {checkpoint}, dataset: {dataset_name}')
        # checkpoint = os.path.join('/data/avishnevskiy/experiments/', checkpoint) 

        try: 
            main(checkpoint, dataset_name, tokenizer_name, generation_name, only_last)
        except OSError:
            checkpoint = os.path.join(checkpoint, 'LATEST')
            main(checkpoint, dataset_name, tokenizer_name, generation_name, only_last)
        except:
            continue
