import pandas as pd
import os
import openai
from utils.data import load_prompts
import argparse
from tqdm import tqdm
from openai.error import InvalidRequestError
from utils.data import Tasks

def make_prompt(source_prompts,source_instruction,target_prompts,target_instruction,k, task_demo='',demo=False):
    demonstration_part=''
    if demo:
        demonstration_part='Definition: '+source_instruction+'\n'+"\n".join(source_prompts[:k])+'\n'
    input_part='Definition: '+target_instruction+'\n'+task_demo+target_prompts
    
    return demonstration_part+input_part

def run_few_shot_sim(source_dataset_name,target_dataset_name,k=1,method='totally-cross-sim', main_dir = 'results'):


    ###### set openai api key #####
    os.environ['AZURE_OPENAI_ENDPOINT'] = "https://kglm.openai.azure.com/"
    os.environ['AZURE_OPENAI_KEY'] = "####" #use your key
    deployment_name='kglm-text-davinci-003'
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future

    model_name = deployment_name.replace("-","_")

    ###########

    result_dir = os.path.join(main_dir, f"dataset_name={target_dataset_name}")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    ############ load dataset ############
    
    target_dataset = load_prompts(target_dataset_name) 
    target_instructions=target_dataset['instruction-target']
    
    if 'cross' in method:
        demo=True
        prompts_dataset=target_dataset['all_possible_pairs'][source_dataset_name]
    else:
        demo=False
        prompts_dataset=target_dataset['all_possible_pairs']['sst2']#sst2 will not be used its a placeholder as load_prompts needs a valid target_dataset_name
    source_instructions=prompts_dataset['instruction-source'] 
    a=Tasks()    
    if 'in' in method:
        task_demo=a.get_one_example(target_dataset_name)
    else:
        task_demo=''
        
    ############

    

    ids=[]
    prompts = []
    preds=[]
    gold_labels=[]
    
    
    for data in tqdm(prompts_dataset['prompts'],desc=f"Evaluating {target_dataset_name} with Source {source_dataset_name}"):
        few_shot_prompt= make_prompt(
            source_prompts=data['source-demonstrations'],
            source_instruction=source_instructions,
            target_prompts=data['target-input'],
            target_instruction=target_instructions,
            k=k,
            task_demo=task_demo,
            demo=demo)
        
        #######
        
        try:
            response = openai.Completion.create(engine=deployment_name, prompt=few_shot_prompt, max_tokens=10, temperature = 0.0, stop=['\n'])
            prediction = response['choices'][0]['text']
        except InvalidRequestError:
            
            prediction='<Junk-answer>'
        #######
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
    parser.add_argument('--source_dataset_name',required=True, type=str)
    parser.add_argument('--target_dataset_name',required=True, type=str)
    parser.add_argument('--method',default='totally-cross-sim', type=str)
    parser.add_argument('--k', default=1,type=int)
    args = parser.parse_args()    
    run_few_shot_sim(args.source_dataset_name,args.target_dataset_name,args.k,args.method)