import pandas as pd
import os
import openai
from utils.data import load_prompts
import argparse
from tqdm import tqdm
from openai.error import InvalidRequestError
from utils.data import Tasks
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

def call_model(prompt,model,tokenizer,device,max_new_tokens=10,model_max_length=2048):
        
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length          
    
    inpts = tokenizer(prompt, return_tensors="pt",truncation=True,max_length= max_inpt_tokens-max_new_tokens).to(device)
    #print(len(inpts.input_ids[0]))
    gen = model.generate(input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):], attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):], pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    #gen = model.generate(input_ids=inpts.input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, num_beams=1, do_sample=False)
    text = tokenizer.decode(gen[0])
    actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])
    #actual_prompt = tokenizer.decode(inpts.input_ids[0])
    pred = text[len(actual_prompt):]
    if pred.startswith("\n\n"):
        pred = pred[2:]
    pred = pred.split("\n")[0]
    return pred, text

def get_combined_source_dataset(prompts_in_dataset,source_dataset_names,target_instruction):

    input=[]
    
    s1,s2,s3,s4=prompts_in_dataset[source_dataset_names[0]],prompts_in_dataset[source_dataset_names[1]],prompts_in_dataset[source_dataset_names[2]],prompts_in_dataset[source_dataset_names[3]]
    
    s1_i,s2_i,s3_i,s4_i=s1['instruction-source'],s2['instruction-source'],s3['instruction-source'],s4['instruction-source']
    s1_p,s2_p,s3_p,s4_p=s1['prompts'],s2['prompts'],s3['prompts'],s4['prompts']
    for t1,t2,t3,t4 in zip(s1_p,s2_p,s3_p,s4_p):
        
        t1_d='Definition: '+s1_i+'\n'+t1['source-demonstrations'][0]
        t2_d='Definition: '+s2_i+'\n'+t2['source-demonstrations'][0]
        t3_d='Definition: '+s3_i+'\n'+t3['source-demonstrations'][0]
        t4_d='Definition: '+s4_i+'\n'+t4['source-demonstrations'][0]
        
        target_input='Definition: '+target_instruction+'\n'+t1['target-input']
        
        inp=t1_d+'\n'+t2_d+'\n'+t3_d+'\n'+t4_d+'\n'+target_input
        
        input.append({
            'input':inp,
            'id':t1['id'],
            'label':t1['label']                        
        })
    return input
        
        
        

def run_few_shot_sim(source_dataset_name,target_dataset_name,model_name,device='cuda:2',k=1,method='totally-cross-sim', main_dir = 'results'):


    ###### Load model #####

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        #output_hidden_states = args.eval_method=='sim_cross_def_label'
        device_map=device,#device
        
    ).eval()
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    generate = lambda prompt, max_new_tokens: call_model(prompt, model=model, tokenizer=tokenizer, device=device)
    model_name = model_name.replace("/","_")
    ###########
    
    source_dataset_names=source_dataset_name.split(',')
    print(target_dataset_name)
    print(source_dataset_names)
    k=len(source_dataset_names)
    print()
    assert k==4
    result_dir = os.path.join(main_dir, f"dataset_name={target_dataset_name}")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    ############ load dataset ############
        
    target_dataset = load_prompts(target_dataset_name)
    demo=True
    task_demo=''
    target_instructions=target_dataset['instruction-target']
    prompts_in_dataset=target_dataset['all_possible_pairs']
    prompts_dataset=get_combined_source_dataset(prompts_in_dataset,source_dataset_names,target_instructions)
        
    a=Tasks()      
    
    ############

    

    ids=[]
    prompts = []
    preds=[]
    gold_labels=[]
    
    
    for data in tqdm(prompts_dataset,desc=f"Evaluating {target_dataset_name} with Source {source_dataset_name}"):
        few_shot_prompt= data['input']
        #print(few_shot_prompt)
        
        #######

        prediction,response=generate(few_shot_prompt, max_new_tokens=15)

        ids.append(data['id'])
        if type(data['label']) is list:
            gold_labels.append(data['label'][0])
        else:
            gold_labels.append(data['label'])
        prompts.append(few_shot_prompt)
        preds.append(prediction)
        #break
    
    eval_dic=dict()
    eval_dic['id']=ids
    eval_dic['prompt']=prompts
    eval_dic['pred']=preds
    eval_dic['true_label']=gold_labels
    
    result_path= os.path.join(result_dir, f"sourcce_dataset={source_dataset_name}-model={model_name}-method={method}-shots={k}.csv")
    result_df=pd.DataFrame(eval_dic)
    result_df.to_csv(result_path,index=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_name',default='sst2,ag_news,mnli,race', type=str)
    parser.add_argument('--target_dataset_name',default='financial_phrasebank', type=str)
    parser.add_argument('--method',default='aba-mixed-cross-sim', type=str)
    parser.add_argument('--model_name',default='/home/eshaan/models/Llama-2-13b-hf/', type=str)
    parser.add_argument('--device',required=True,type=str)
    parser.add_argument('--k', default=1,type=int)
    args = parser.parse_args()    
    run_few_shot_sim(args.source_dataset_name,args.target_dataset_name,args.model_name,args.device,args.k,args.method)

