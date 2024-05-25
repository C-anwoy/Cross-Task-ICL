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


def force_decode(prompt, label_space, model, tokenizer, device):
    model.to(device)

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Make a forward pass to get the logits
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    label_probs = []

    for label in label_space:
        label_id = tokenizer.encode(label)[1]
        prob_label = probs[0, -1, label_id].item()
        label_probs.append(prob_label)

    
    label_probs = np.array(label_probs)
    output = label_space[np.argmax(label_probs)]
    return output

def run(llm_name, device, SOURCE_TASKS, TARGET_TASKS, prompter=Prompt()):

    if llm_name == 'llama-2-7b':
        llama_2_7b_path = '/home/models/Llama-2-7b-hf'
        llm_model = AutoModelForCausalLM.from_pretrained(llama_2_7b_path, torch_dtype=torch.float16, trust_remote_code=True, device_map = device).eval()
        tokenizer = AutoTokenizer.from_pretrained(llama_2_7b_path)
    
    elif llm_name == 'llama-2-13b':
        llama_2_13b_path = '/home/eshaan/models/Llama-2-13b-hf/'
        llm_model = AutoModelForCausalLM.from_pretrained(llama_2_13b_path, torch_dtype=torch.float16, trust_remote_code=True, device_map = device).eval()
        tokenizer = AutoTokenizer.from_pretrained(llama_2_13b_path)


    dir = 'results/force_decode'
    
    for target_task in TARGET_TASKS:

        with open(f'data/target/{target_task}.pkl','rb') as f:
            target_set=pickle.load(f)
    
        target_instruction = target_set['instruction']
        target_prompt_id = target_set['prompt_temp_id']
        target_label_space = target_set['labels']

        for source_task in SOURCE_TASKS:
                
                with open(f'data/source/{source_task}_train.pkl','rb') as f:
                    source_data = pickle.load(f)

                result_dir = os.path.join(dir, f'{target_task}')
                result_path= os.path.join(result_dir, f"source_dataset={source_task}-model={llm_name}-shots={1}.csv")
                
                source_instruction = source_data['instruction']

                if not os.path.isfile(result_path):

                    ids=[]
                    prompts = []
                    preds=[]
                    gold_labels=[]
                    correct = []

                    for i in range(len(target_set['data'])):

                        datapoint = target_set['data'][i]

                        context = datapoint['sim_examples'][source_task]
                        context = context.strip()
                        context = f'Definition: {source_instruction} \n ' + context
                        input_ex = prompter.get_input_data_with_def(target_instruction, datapoint, target_prompt_id)
                        input_ex = input_ex.strip()


                        input_prompt = context + '\n' + input_ex
                        #prediction = get_black_box_model_output(input_prompt, llm_name, llm_model, tokenizer, device)
                        prediction = force_decode(input_prompt, target_label_space, llm_model, tokenizer, device)

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
        
                    result_path= os.path.join(result_dir, f"source_dataset={source_task}-model={llm_name}-shots={1}.csv")
                    result_df=pd.DataFrame(eval_dic)
                    result_df.to_csv(result_path,index=False)
            
        print(f'{target_task} done !')

    

if __name__ == "__main__":

    SOURCE_TASKS = ['ARC-Easy', 'ag_news', 'boolq', 'commonsense_qa', 'conll2003_pos', 'conll2003_ner', 'mnli', 'qqp', 'race', 'sst2']
    TARGET_TASKS = ['ARC-Challenge', 'financial_phrasebank', 'medmcqa', 'sciq', 'social_i_qa']

    parser = argparse.ArgumentParser()

    parser.add_argument('--llm_name', required=True, type=str)
    parser.add_argument('--device', required=True, type=str)
    args = parser.parse_args()    
    
    run(args.llm_name, args.device, SOURCE_TASKS, TARGET_TASKS)