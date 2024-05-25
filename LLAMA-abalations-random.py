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

example_dict={'ag_news' : {0: 'Definition: Given a sentence do text classification, the sentence is a clipping from a news article that may be either related to sports, business, technology, or world news. You are to recognize the category of the sentence and label them as "sports", "business", "technology" or "world" news \n Sentence: First class to the moon London - British airline magnate Richard Branson announced a plan on Monday for the world #39;s first commercial space flights, saying  quot;thousands quot; of fee-paying astronauts could be sent into orbit in the near future. \n Label: technology',
            	1: 'Definition: Given a sentence do text classification, the sentence is a clipping from a news article that may be either related to sports, business, technology, or world news. You are to recognize the category of the sentence and label them as "sports", "business", "technology" or "world" news \n Sentence: U.S. Economy Grows at Slower Pace Than Expected Struggling under the weight of a bloated trade deficit, the U.S. economy grew at a relatively modest 2.8 percent annual rate in the second quarter. \n Label: business',
            	2: 'Definition: Given a sentence do text classification, the sentence is a clipping from a news article that may be either related to sports, business, technology, or world news. You are to recognize the category of the sentence and label them as "sports", "business", "technology" or "world" news \n Sentence: Sri Lanka beat Pakistan in Faisalabad Test (AFP) AFP - Sri Lanka defeated Pakistan by 201 runs in the first cricket Test here to take a 1-0 lead in the two-match series. \n Label: sports'}


            ,'boolq' : {0: 'Definition: Given a context and a question do binary true and false type text classification. You are given a passage as context and a question related to the passage that can be answered as "True" or "False". Based on the context, question and your reasoning ability answer in a "True" and "False". \n Context: Firearms regulations are uniform throughout Florida, and a carry license is valid everywhere other than in a few specially-defined areas. These prohibited places include any police station, prison, courthouse, polling place, government meeting place, airport, seaport, or tavern. Concealed carry is also prohibited in any school, except for authorized security personnel or armed marshals. \n Question: can you carry a concealed weapon in florida \n Answer: true',
                2: 'Definition: Given a context and a question do binary true and false type text classification. You are given a passage as context and a question related to the passage that can be answered as "True" or "False". Based on the context, question and your reasoning ability answer in a "True" and "False". \n Context: A timing belt, timing chain or cambelt is a part of an internal combustion engine that synchronizes the rotation of the crankshaft and the camshaft(s) so that the engine\'s valves open and close at the proper times during each cylinder\'s intake and exhaust strokes. In an interference engine the timing belt or chain is also critical to preventing the piston from striking the valves. A timing belt is usually a toothed belt -- a drive belt with teeth on the inside surface. A timing chain is a roller chain. \n Question: is timing belt and cam belt the same \n Answer: true',
                1: 'Definition: Given a context and a question do binary true and false type text classification. You are given a passage as context and a question related to the passage that can be answered as "True" or "False". Based on the context, question and your reasoning ability answer in a "True" and "False". \n Context: In 2010, the South Korean government legalized dual citizenship for some South Koreans who have acquired another nationality/citizenship, as well as foreigners who lived in South Korea for five years (two years if married to a South Korean). \n Question: can you have dual citizenship in us and korea \n Answer: true'}


,           'commonsense_qa' : {0: 'Definition: The following task relates to commonsense reasoning. It consists of a question that can be easily solved using logical abilities and reasoning, a set of five options  "A.", "B.", "C.", "D." and "E." are also provided along with the question, one of these options answers the question logically. Use your reasoning ability to select the most appropriate answer from the provided choices "A.", "B.", "C.", "D." and "E." and assign these choices (i.e  "A.", "B.", "C.", "D." and "E.") as the label \n Question: The atoms that make up oxygen gas\nA. share two covalent bonds.\nB. have a definite shape.\nC. share a pair of neutrons.\nD. have two free electrons \n Answer: A',
                1: 'Definition: The following task relates to commonsense reasoning. It consists of a question that can be easily solved using logical abilities and reasoning, a set of five options  "A.", "B.", "C.", "D." and "E." are also provided along with the question, one of these options answers the question logically. Use your reasoning ability to select the most appropriate answer from the provided choices "A.", "B.", "C.", "D." and "E." and assign these choices (i.e  "A.", "B.", "C.", "D." and "E.") as the label \n Question: In New York State, an observer will usually see the Sun rise in the\nA. north\nB. south\nC. east\nD. west \n Answer: C',
                2 : 'Definition: The following task relates to commonsense reasoning. It consists of a question that can be easily solved using logical abilities and reasoning, a set of five options  "A.", "B.", "C.", "D." and "E." are also provided along with the question, one of these options answers the question logically. Use your reasoning ability to select the most appropriate answer from the provided choices "A.", "B.", "C.", "D." and "E." and assign these choices (i.e  "A.", "B.", "C.", "D." and "E.") as the label \n Question: Which of these living things in a grassland ecosystem depend on the presence of a variety of bacteria and other microorganisms in the soil?\nA. worms only\nB. green plants only\nC. plants and animals only\nD. all the organisms in the ecosystem \n Answer: D'}

,"mnli" : {0: 'Definition: Given Sentence 1 which is a premise and Sentence 2 which is a hypothesis do natural language inference on the pair. In natural language inference we mark whether the premise and hypothesis are "neutral", "contradiction" or "entailment". The pair are said to be "entailed" if the premise justifies/supports the hypothesis, if the pair contradict each other we label them as "contradiction" and label them "neutral" in all other cases \n Premise: He started slowly back to the bunkhouse. \n Hypothesis: He returned slowly to the bunkhouse. \n Label: entailment',
            	1: 'Definition: Given Sentence 1 which is a premise and Sentence 2 which is a hypothesis do natural language inference on the pair. In natural language inference we mark whether the premise and hypothesis are "neutral", "contradiction" or "entailment". The pair are said to be "entailed" if the premise justifies/supports the hypothesis, if the pair contradict each other we label them as "contradiction" and label them "neutral" in all other cases \n Premise: Poirot, I exclaimed, with relief, and seizing him by both hands, I dragged him into the room. \n Hypothesis: Poirot was now back and I was sorry that he would take over what I now considered my own investigation. \n Label: contradiction',
            	2: 'Definition: Given Sentence 1 which is a premise and Sentence 2 which is a hypothesis do natural language inference on the pair. In natural language inference we mark whether the premise and hypothesis are "neutral", "contradiction" or "entailment". The pair are said to be "entailed" if the premise justifies/supports the hypothesis, if the pair contradict each other we label them as "contradiction" and label them "neutral" in all other cases \n Premise: Analyzing Postal Service accounts for depreciation, fuel, and maintenance for city delivery carriers, we have estimated the average city delivery vehicle cost per route. \n Hypothesis: Driving cost estimates can be averaged will sufficient data. \n Label: neutral'}


,'qqp' : {0: 'Definition: Given two question pairs do text classification based on whether they are duplicates or not. The questions are mined from the popular online discussion forum Quora. As duplicate question might be present on Quora, the task is to label two identical questions as "duplicate" if they ask the same query else label the pair as "not duplicate". \n Question 1: What is the district of Edgware and how does the lifestyle compare to the London Borough of Islington? \n Question 2: What is the county of Edgware and how does the lifestyle compare to the London Borough of Enfield? \n Label: not duplicate',
            	1: 'Definition: Given two question pairs do text classification based on whether they are duplicates or not. The questions are mined from the popular online discussion forum Quora. As duplicate question might be present on Quora, the task is to label two identical questions as "duplicate" if they ask the same query else label the pair as "not duplicate". \n Question 1: What are the best books on cosmology? \n Question 2: Which is the best book for cosmology? \n Label: duplicate',
            	2: 'Definition: Given two question pairs do text classification based on whether they are duplicates or not. The questions are mined from the popular online discussion forum Quora. As duplicate question might be present on Quora, the task is to label two identical questions as "duplicate" if they ask the same query else label the pair as "not duplicate". \n Question 1: Why do we, as human beings, use water for? \n Question 2: What do we use water for? \n Label: duplicate'}
}

print(example_dict['ag_news'][0])
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

def get_combined_source_dataset(prompts_in_dataset,source_dataset_names,target_instruction,seed):

    input=[]
    
    s1=prompts_in_dataset[source_dataset_names]
    s1_p=s1['prompts']
    
    dem=example_dict[source_dataset_names][seed]
    for t1 in s1_p:        
        target_input='Definition: '+target_instruction+'\n'+t1['target-input']
        
        inp=dem+target_input
        
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
    result_dir = os.path.join(main_dir, f"dataset_name={target_dataset_name}")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    ############ load dataset ############
        
    target_dataset = load_prompts(target_dataset_name)
    demo=True
    task_demo=''
    target_instructions=target_dataset['instruction-target']
    prompts_in_dataset=target_dataset['all_possible_pairs']
    prompts_dataset=get_combined_source_dataset(prompts_in_dataset,source_dataset_name,target_instructions,args.seed)
        
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
    
    result_path= os.path.join(result_dir, f"sourcce_dataset={source_dataset_name}-model={model_name}-method={method}{args.seed}-shots={k}.csv")
    result_df=pd.DataFrame(eval_dic)
    result_df.to_csv(result_path,index=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_name',default='sst2,ag_news,mnli,race', type=str)
    parser.add_argument('--target_dataset_name',default='financial_phrasebank', type=str)
    parser.add_argument('--method',default='aba-mixed-cross-sim', type=str)
    parser.add_argument('--model_name',default='/home/eshaan/models/Llama-2-13b-hf/', type=str)
    parser.add_argument('--device',required=True,type=str)
    parser.add_argument('--seed', default=0,type=int)
    parser.add_argument('--k', default=1,type=int)
    args = parser.parse_args()    
    run_few_shot_sim(args.source_dataset_name,args.target_dataset_name,args.model_name,args.device,args.k,args.method)

