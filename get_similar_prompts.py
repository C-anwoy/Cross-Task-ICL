from utils.data import load_dataset
from utils.data import Prompt
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
import pickle

def main(args):
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
    target_tasks=[
            'ARC-Challenge',
            'financial_phrasebank',
            'medmcqa',
            'sciq',
            'social_i_qa'
            ]
    prompter=Prompt()
    
    if args.make_psudo:
        target_name='target-unlabeled'
        type='train'
        path='data/target-psudo-prompts'
    else:
        target_name='target'
        type='test'
        path='data/target-prompts'
    
    for t in tqdm(target_tasks):
        target_data=load_dataset(target_name,t,type)
        target_id=target_data['prompt_temp_id']
        target_dataset=target_data['data']
        target_inst=target_data['instruction']
        
        prompt4targets={}
        
        for s in source_tasks:
            source_data=load_dataset('source',s,'train')        
            source_id=source_data['prompt_temp_id']
            source_dataset=source_data['data']
            source_inst=source_data['instruction']
            
            source_dataset_prompts=[(prompter.get_input_data_without_def(d,source_id)+" "+str(d['label'])) for d in source_dataset]
            source_dataset_embs=torch.stack([d['emb'] for d in source_dataset],dim=0).to(args.device)
            
            prompts=[]
            for data in target_dataset:
                
                t_data=data.copy()                
                t_e=t_data['emb']
                t_e = torch.unsqueeze(t_e, dim=0).to(args.device) 
                t_p=prompter.get_input_data_without_def(t_data,target_id)               
                sim=F.cosine_similarity(t_e,source_dataset_embs).detach().cpu()
                top_k=torch.argsort(sim,descending=True)[:args.k].numpy()
                entries=[source_dataset_prompts[k] for k in top_k]
                t_data['source-demonstrations']=entries
                t_data['target-input']=t_p
                
                prompts.append(t_data)
            
            prompt4targets[s]={
                'instruction-source':source_inst,
                'prompts':prompts
            }
        data_to_save={
            'dataset_name':t,
            'instruction-target':target_inst,
            'all_possible_pairs':prompt4targets
        }
        
        
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/{t}_prompts_k={args.k}.pkl', 'wb') as openfile:
        # Reading from json file
            pickle.dump(data_to_save, openfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0', type=str)
    parser.add_argument('--make_psudo',action='store_true')
    parser.add_argument('--k',default=4, type=int)
    args = parser.parse_args()    
    main(args)
    
