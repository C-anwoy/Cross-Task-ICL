import torch
import os
import openai
import pickle
from utils.data import Prompt
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def call_model(prompt,model,tokenizer,device,max_new_tokens=10,model_max_length=2048):
        
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length          
    
    inpts = tokenizer(prompt, return_tensors="pt",truncation=True,max_length= max_inpt_tokens-max_new_tokens).to(device)
    #print(len(inpts.input_ids[0]))
    gen = model.generate(input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):], attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):], pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    #gen = model.generate(input_ids=inpts.input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, num_beams=1, do_sample=False)
    text = tokenizer.decode(gen[0])
    actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])
    pred = text[len(actual_prompt):]
    if pred.startswith("\n\n"):
        pred = pred[2:]
    pred = pred.split("\n")[0]
    return pred, text

def get_llm_emb(text, llm_model, llm_tokenizer, device):
    batch_tokens = llm_tokenizer(text, return_tensors="pt").to(device)
    llm_model = llm_model.to(device)

    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim] from last layer
        hidden_state = llm_model(**batch_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]

    hidden_state = hidden_state.squeeze()
    last_token_emb = hidden_state[-1]
    mean_emb = torch.mean(hidden_state, dim =0)

    return mean_emb, last_token_emb

def create_embedding(data, model):
    o=model.encode(data, convert_to_tensor=True).detach().cpu()
    return o

def get_similar_example(source_name, source_instruction, target_example, target_prompt_id, device):
    
    with open(f'data/source_emb/{source_name}_emb.pkl','rb') as f:
        source_emb = pickle.load(f)
    
    with open(f'data/source/{source_name}_train.pkl','rb') as f:
        source_set = pickle.load(f)

    prompter = Prompt()

    source_data = source_set['data']
    source_prompt_id = source_set['prompt_temp_id']

    model = SentenceTransformer("all-MiniLM-L6-v2")

    target = [prompter.get_input_data_without_def(target_example, target_prompt_id)]
    target_emb = create_embedding(target, model.to(device))

    sim = F.cosine_similarity(source_emb, target_emb, dim=1)
    index = torch.argmax(sim).item()

    source_example = prompter.get_input_data_with_def(source_instruction, source_data[index], source_prompt_id) + ' ' + source_data[index]['label']
    return source_example
    


def get_black_box_model_output(input_prompt, black_box_model_name, bb_model=None, bb_tokenizer=None, bb_device=None):

    if black_box_model_name in ['llama-2-7b', 'llama-2-13b']:

        generate = lambda prompt, max_new_tokens: call_model(prompt, model=bb_model, tokenizer=bb_tokenizer, device=bb_device, max_new_tokens=max_new_tokens)
        prediction, _ = generate(input_prompt, max_new_tokens=10)

        return prediction

    elif black_box_model_name in ['llama-2-7b-chat', 'llama-2-13b-chat']:
        generate = lambda prompt, max_new_tokens: call_model(prompt, model=bb_model, tokenizer=bb_tokenizer, device=bb_device, max_new_tokens=max_new_tokens)
        prediction, _ = generate(input_prompt, max_new_tokens=10)

        return prediction
    
    elif black_box_model_name=='gpt3':
        os.environ['AZURE_OPENAI_ENDPOINT'] = "https://kglm.openai.azure.com/"
        os.environ['AZURE_OPENAI_KEY'] = "####" #give api key
        deployment_name='kglm-text-davinci-003'
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15' # this may change in the future

        try:
            response = openai.Completion.create(engine=deployment_name, prompt=input_prompt, max_tokens=10, temperature = 0.0, stop=['\n'])
            prediction = response['choices'][0]['text']
        except openai.error.InvalidRequestError:
            prediction='<Junk-answer>'

        
        return prediction


def compare_output(output, gold_label):
    
    output= str(output).replace('Label:','')
    output = output.strip(' .[]":\'').lower()
    output=output.replace('.','')
    output= output.split(' ')[0]
    
    if type(gold_label) is list:

        for g in gold_label:
            g = str(g).lower()
            if (output==g):
                return 1
            
        return 0

    else:
        gold_label = str(gold_label).lower()
        if (output==gold_label):
            return 1
        else:
            return 0