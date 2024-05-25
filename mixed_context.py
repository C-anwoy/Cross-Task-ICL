import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import torch.nn.functional as F
import numpy as np
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import pandas as pd
from utils.data import Prompt
from utils.llm import *
import argparse
from openai.error import InvalidRequestError
import tiktoken

def run(llm_name, device, SOURCE_TASKS, TARGET_TASKS, cross_shots=1, k=8, prompter=Prompt()):

    if llm_name == 'llama-2-7b':
        llama_2_7b_path = 'meta-llama/Llama-2-7b-hf'
        llm_model = AutoModelForCausalLM.from_pretrained(llama_2_7b_path, torch_dtype=torch.float16, trust_remote_code=True, device_map = device).eval()
        tokenizer = AutoTokenizer.from_pretrained(llama_2_7b_path)
    
    elif llm_name == 'llama-2-13b':
        llama_2_13b_path = 'meta-llama/Llama-2-13b-hf'
        llm_model = AutoModelForCausalLM.from_pretrained(llama_2_13b_path, torch_dtype=torch.float16, trust_remote_code=True, device_map = device).eval()
        tokenizer = AutoTokenizer.from_pretrained(llama_2_13b_path)
    
    elif llm_name == 'gpt3':
        llm_model=None
        tokenizer=None


    dir = 'results/mixed_context_exp'

    best_source_tasks = {'llama-2-7b': {'ARC-Challenge': ['commonsense_qa', 'ARC-Easy', 'race', 'qqp', 'boolq', 'mnli', 'conll2003_ner', 'conll2003_pos'], 
                         'financial_phrasebank': ['sst2', 'mnli', 'ag_news', 'race', 'qqp', 'ARC-Easy', 'boolq', 'commonsense_qa'], 
                         'medmcqa': ['commonsense_qa', 'race', 'ARC-Easy', 'boolq', 'ag_news', 'conll2003_pos', 'qqp', 'conll2003_ner'], 
                         'sciq': ['commonsense_qa', 'race', 'ARC-Easy', 'boolq', 'ag_news', 'mnli', 'sst2', 'conll2003_pos'], 
                         'social_i_qa': ['race', 'commonsense_qa', 'ARC-Easy', 'boolq', 'ag_news', 'qqp', 'sst2', 'conll2003_ner']},

                         'llama-2-13b': {'ARC-Challenge': ['ARC-Easy', 'commonsense_qa', 'race', 'boolq', 'qqp', 'sst2', 'ag_news', 'conll2003_pos'], 
                         'financial_phrasebank': ['qqp', 'mnli', 'ag_news', 'sst2', 'ARC-Easy', 'race', 'boolq', 'commonsense_qa'], 
                         'medmcqa': ['race', 'commonsense_qa', 'ARC-Easy', 'boolq', 'ag_news', 'conll2003_ner', 'conll2003_pos', 'sst2'], 
                         'sciq': ['commonsense_qa', 'ARC-Easy', 'race', 'boolq', 'conll2003_ner', 'mnli', 'conll2003_pos','ag_news'], 
                         'social_i_qa': ['race', 'commonsense_qa', 'ARC-Easy', 'conll2003_pos', 'boolq', 'conll2003_ner', 'ag_news', 'qqp']},

                         'gpt3': {'ARC-Challenge': ['race', 'boolq', 'ARC-Easy', 'commonsense_qa', 'mnli', 'ag_news', 'qqp', 'sst2'], 
                         'financial_phrasebank': ['ag_news', 'commonsense_qa', 'ARC-Easy', 'boolq', 'sst2', 'qqp', 'race', 'mnli'], 
                         'medmcqa': ['boolq', 'ARC-Easy', 'qqp', 'ag_news', 'commonsense_qa',  'sst2', 'race', 'conll2003_pos'], 
                         'sciq': ['race', 'ARC-Easy', 'commonsense_qa', 'boolq', 'ag_news', 'qqp', 'sst2', 'mnli'], 
                         'social_i_qa': ['race', 'ARC-Easy', 'ag_news', 'mnli', 'commonsense_qa',  'sst2', 'boolq', 'conll2003_ner']}
                        }
    
    # for LLaMA-7B
    # best_source_tasks = {'ARC-Challenge': ['commonsense_qa', 'ARC-Easy', 'race', 'qqp', 'boolq', 'mnli', 'conll2003_ner', 'conll2003_pos'], 
    #                      'financial_phrasebank': ['sst2', 'mnli', 'ag_news', 'race', 'qqp', 'ARC-Easy', 'boolq', 'commonsense_qa'], 
    #                      'medmcqa': ['commonsense_qa', 'race', 'ARC-Easy', 'boolq', 'ag_news', 'conll2003_pos', 'qqp', 'conll2003_ner'], 
    #                      'sciq': ['commonsense_qa', 'race', 'ARC-Easy', 'boolq', 'ag_news', 'mnli', 'sst2', 'conll2003_pos'], 
    #                      'social_i_qa': ['race', 'commonsense_qa', 'ARC-Easy', 'boolq', 'ag_news', 'qqp', 'sst2', 'conll2003_ner']}
    
    #for LLaMA-13B
    # best_source_tasks = {'ARC-Challenge': ['ARC-Easy', 'commonsense_qa', 'race', 'boolq', 'qqp', 'sst2', 'ag_news', 'conll2003_pos'], 
    #                      'financial_phrasebank': ['qqp', 'mnli', 'ag_news', 'sst2', 'ARC-Easy', 'race', 'boolq', 'commonsense_qa'], 
    #                      'medmcqa': ['race', 'commonsense_qa', 'ARC-Easy', 'boolq', 'ag_news', 'conll2003_ner', 'conll2003_pos', 'sst2'], 
    #                      'sciq': ['commonsense_qa', 'ARC-Easy', 'race', 'boolq', 'conll2003_ner', 'mnli', 'conll2003_pos','ag_news'], 
    #                      'social_i_qa': ['race', 'commonsense_qa', 'ARC-Easy', 'conll2003_pos', 'boolq', 'conll2003_ner', 'ag_news', 'qqp']}

    #for gpt3
    # best_source_tasks = {'ARC-Challenge': ['race', 'boolq', 'ARC-Easy', 'commonsense_qa', 'mnli', 'ag_news', 'qqp', 'sst2'], 
    #                      'financial_phrasebank': ['ag_news', 'commonsense_qa', 'ARC-Easy', 'boolq', 'sst2', 'qqp', 'race', 'mnli'], 
    #                      'medmcqa': ['boolq', 'ARC-Easy', 'qqp', 'ag_news', 'commonsense_qa',  'sst2', 'race', 'conll2003_pos'], 
    #                      'sciq': ['race', 'ARC-Easy', 'commonsense_qa', 'boolq', 'ag_news', 'qqp', 'sst2', 'mnli'], 
    #                      'social_i_qa': ['race', 'ARC-Easy', 'ag_news', 'mnli', 'commonsense_qa',  'sst2', 'boolq', 'conll2003_ner']}
    
    for target_task in TARGET_TASKS:

        with open(f'mixed_context_exp/datasets/target/{target_task}.pkl','rb') as f:
            target_set=pickle.load(f)
    
        target_instruction = target_set['instruction']
        target_prompt_id = target_set['prompt_temp_id']
        target_label_space = target_set['labels']

        selected_source = [best_source_tasks[llm_name][target_task][i] for i in range(cross_shots)]

        selected_source_instructions = []
                
        for source_task in selected_source:
            with open(f'data/source/{source_task}_train.pkl','rb') as f:
                source_data = pickle.load(f)
            
            source_instruction = source_data['instruction']
            selected_source_instructions.append(source_instruction)
        
        #for source_task in SOURCE_TASKS:
                
        # with open(f'data/source/{source_task}_train.pkl','rb') as f:
        #     source_data = pickle.load(f)

        result_dir = os.path.join(dir, f'{target_task}')
        result_path= os.path.join(result_dir, f"model={llm_name}-k={k}-cross_shots={cross_shots}.csv")
        
        #source_instruction = source_data['instruction']

        if not os.path.isfile(result_path):

            ids=[]
            prompts = []
            preds=[]
            gold_labels=[]
            correct = []

            for i in range(len(target_set['data'])):

                datapoint = target_set['data'][i]
                context = ''

                #cross-task examples
                for j in range(cross_shots): 
                    ex = datapoint['sim_examples'][selected_source[j]]
                    ex = ex.strip()
                    context = f'Definition: {selected_source_instructions[j]} \n' + ex + '\n' + context
                
                #in-task examples
                in_context = ''

                if (k-cross_shots)!=0:
    
                    in_context_def = f'Definition: {target_instruction} \n'
                
                    for l in range(int(k-cross_shots)):
                        in_context = datapoint['in_task_sim_examples'][l] + '\n' + in_context

                    in_context = in_context_def + '\n' + in_context
                
                input_ex = prompter.get_input_data_without_def(datapoint, target_prompt_id)
                input_ex = input_ex.strip()


                input_prompt = context + '\n' + in_context + '\n' + input_ex
                prediction = get_black_box_model_output(input_prompt, llm_name, llm_model, tokenizer, device)

                ids.append(datapoint['id'])

                if type(datapoint['label']) is list:
                    gold_labels.append(datapoint['label'][0])
                else:
                    gold_labels.append(datapoint['label'])

                prompts.append(input_prompt)
                preds.append(prediction)
                correct.append(compare_output(prediction, datapoint['label']))
        
            eval_dic=dict()
            eval_dic['id']=ids
            eval_dic['prompt']=prompts
            eval_dic['pred']=preds
            eval_dic['true_label']=gold_labels
            eval_dic['correct'] = correct


            result_dir = os.path.join(dir, f'{target_task}')

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            result_path= os.path.join(result_dir, f"model={llm_name}-k={k}-cross_shots={cross_shots}.csv")
            result_df=pd.DataFrame(eval_dic)
            result_df.to_csv(result_path,index=False)
            
        print(f'{target_task} done !')

    

if __name__ == "__main__":

    SOURCE_TASKS = ['ARC-Easy', 'ag_news', 'boolq', 'commonsense_qa', 'conll2003_pos', 'conll2003_ner', 'mnli', 'qqp', 'race', 'sst2']
    TARGET_TASKS = ['ARC-Challenge', 'financial_phrasebank', 'medmcqa', 'sciq', 'social_i_qa']

    parser = argparse.ArgumentParser()

    #parser.add_argument('--llm_name', required=True, type=str)
    parser.add_argument('--device', required=True, type=str)
    #parser.add_argument('--cross_shots', required=True, type=int)
    args = parser.parse_args()
    
    for i in range(9):
        run('gpt3', args.device, SOURCE_TASKS, TARGET_TASKS, i)
        print(f"################## cross-shots = {i} done ######################")