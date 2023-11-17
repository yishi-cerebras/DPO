from evaluation import main
import os


checkpoints = [
    # '/data/avishnevskiy/experiments/dpo_pythia1.4_continuing_epoch3-20231115-072829/LATEST',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'dpo_btlm-20231024-232611', #.3 hh last
    # 'dpo_btlm-20231108-075431', #.01 hh last
    'dpo_btlm-20231027-083432', #.3 shp
    # 'dpo_btlm_shp-20231029-084512', #.5 shp
    # 'same_hp_anthropic_dpo_pythia28_model', #Pythia without LoRa float32 hh last
    # 'same_hp_anthropic_60k', #Pythia without LoRa float32 60k checkpoint hh last
    # 'trl_dpo_pythia_fp16-20231011-010059', #Pythia without LoRa bfloat16 hh last
    # 'authors_pythia_model', #authors model hh last
    # 'sft-btlm', #sft btlm hh last
    # 'sft-btlm-shp', #sft btlm shp
    # 'dpo_btlm-20231108-075431', #btlm-dpo-hh 0.01 shp
    # 'dpo_btlm-20231025-163346', #btlm-dpo-hh 0.5 shp
    # 'sft-btlm', #sft btlm-hh on shp
    'dpo_btlm-20231027-083432', #.3 shp on hh last
    # 'dpo_btlm_shp_hh-20231109-213829', #BTLM-SFT-AHH-DPO-SHP on shp
]

dataset_names = [
    # 'hh',
    # 'hh',
    # 'shp',
    # 'hh',
    # 'hh',
    'shp',
    # 'shp',
    # 'hh',
    # 'hh',
    # 'hh',
    # 'hh',
    # 'hh',
    # 'shp',
    # 'shp',
    # 'shp',
    # 'shp',
    'hh',
    # 'shp'
]

tokenizer_names = [
    # 'EleutherAI/pythia-1.4b',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'EleutherAI/pythia-2.8b',
    # 'EleutherAI/pythia-2.8b',
    # 'EleutherAI/pythia-2.8b',
    # 'EleutherAI/pythia-2.8b',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base',
    'cerebras/btlm-3b-8k-base',
    # 'cerebras/btlm-3b-8k-base'
]

generation_names = [
    # 'pythia_beta0.05-7.5k-checkpoint_512examples',
    # 'btlm_new_hh',
    # 'btlm_new_shp',
    # 'dpo_btlm_0.3_hh_last256',
    # 'dpo_btlm_0.01_hh_last256',
    'dpo_btlm_0.3_shp_all_categories',
    # 'dpo_btlm_0.5_shp_all_categories',
    # 'pythia_without_LoRa_float32_last256',
    # 'pythia_without_LoRa_float32_60k_last256',
    # 'pythia_without_LoRa_bfloat16_last256',
    # 'authors_model_hh_last256',
    # 'sft_btlm_hh_last256',
    # 'sft_btlm_shp_all_categories',
    # 'btlm-dpo-hh_0.01_shp_all_categories',
    # 'btlm-dpo-hh_0.5_shp_all_categories',
    # 'sft_btlm-hh_shp_all_categories',
    'dpo_btlm_0.3_shp_hh_last256',
    # 'BTLM-SFT-AHH-DPO-SHP_on_shp_all_categories'
]

only_lasts = [
    # True, 
    # True,
    # False,
    # False, 
    # True,
    True,
    # True,
    # True,
    # True,
    # False,
    # False,
    # False,
    # False,
    True,
    # False
]

if __name__ == "__main__":
    for checkpoint, dataset_name, tokenizer_name, generation_name, only_last in zip(checkpoints, dataset_names, tokenizer_names, generation_names, only_lasts):
        print(f'generating for checkpoint: {checkpoint}, dataset: {dataset_name}')
        checkpoint = os.path.join('/data/avishnevskiy/experiments/', checkpoint) 

        try: 
            main(checkpoint, dataset_name, tokenizer_name, generation_name, only_last)
        except OSError:
            checkpoint = os.path.join(checkpoint, 'LATEST')
            main(checkpoint, dataset_name, tokenizer_name, generation_name, only_last)
        except:
            continue
