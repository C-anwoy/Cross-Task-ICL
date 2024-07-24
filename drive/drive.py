import os
import time

target_tasks=[
              'ARC-Challenge',
              'financial_phrasebank',
              'medmcqa',
              'sciq',
              'social_i_qa'
              ]
source_tasks=[
            'ag_news',
            'ARC-Easy',
            'boolq',
            'commonsense_qa',
            'conll2003_pos',
            'conll2003_ner',
            'mnli',
            'qqp',
            'race',
            'sst2'
            ]

model_name='path_to_Llama-2-model'
# totally-cross-sim zero-shot -> table 1
# in-n-cross-sim in-task -> tabel 4
for target_task in target_tasks:
    for s in source_tasks:
        
        cmd_line=f"python LLAMA.py --source_dataset_name {s} --target_dataset_name {target_task} --method totally-cross-sim --model_name {model_name} --k 1 --device cuda:1"
        print(cmd_line)
        ret_status = os.system(cmd_line)
        if ret_status != 0:
            print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
            exit()
for target_task in target_tasks:
    for s in source_tasks:
        
        cmd_line=f"python LLAMA.py --source_dataset_name {s} --target_dataset_name {target_task} --method in-n-cross-sim --model_name {model_name} --k 1 --device cuda:1"
        print(cmd_line)
        ret_status = os.system(cmd_line)
        if ret_status != 0:
            print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
            exit()
for target_task in target_tasks:
    for s in source_tasks:
        
        cmd_line=f"python LLAMA.py --source_dataset_name {s} --target_dataset_name {target_task} --method in-task --model_name {model_name} --k 0 --device cuda:1"
        print(cmd_line)
        ret_status = os.system(cmd_line)
        if ret_status != 0:
            print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
            exit()
for target_task in target_tasks:
    for s in source_tasks:
        
        cmd_line=f"python LLAMA.py --source_dataset_name {s} --target_dataset_name {target_task} --method zero-shot --model_name {model_name} --k 0 --device cuda:2"
        print(cmd_line)
        ret_status = os.system(cmd_line)
        if ret_status != 0:
            print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
            exit()
