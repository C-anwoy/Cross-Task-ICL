import json
import pickle

def load_dataset(benchmark,dataset_name,set):   
    path=f'data/{benchmark}/{dataset_name}_{set}.pkl'          
    with open(path,'rb') as f:
        test_set=pickle.load(f)
    return test_set

def load_prompts(dataset_name,path_d='target-prompts',k=4):
    path=f'data/{path_d}/{dataset_name}_prompts_k={k}.pkl'
    with open(path,'rb') as f:
        test_set=pickle.load(f)
    return test_set
    

class Prompt:
    def __init__(self):
        
        self.prompt_temps_dic={
        '001':'Definition: {} \nPremise: {} \nHypothesis: {} \nLabel:',
        '002':'Definition: {} \nQuestion 1: {} \nQuestion 2: {} \nLabel:',
        '003':'Definition: {} \nContext: {} \nQuestion: {} \nAnswer:',
        '004':'Definition: {} \nSentence: {} \nLabel:',
        '005':'Definition: {} \nQuestion: {} \nAnswer:',
        '006':'Definition: {} \nSentence 1: {} \nSentence 2: {} \nLabel:',
        '007':'Definition: {} \nQuestion: {} \nSentence: {} \nLabel:',
        }

        self.prompt_temps_dic_without_def={
            '001':'Premise: {} \nHypothesis: {} \nLabel:',
            '002':'Question 1: {} \nQuestion 2: {} \nLabel:',
            '003':'Context: {} \nQuestion: {} \nAnswer:',
            '004':'Sentence: {} \nLabel:',
            '005':'Question: {} \nAnswer:',
            '006':'Sentence 1: {} \nSentence 2: {} \nLabel:',
            '007':'Question: {} \nSentence: {} \nLabel:'
        }
        
    def get_input_data_with_def(self,instructions,data,prompt_id):         

        promt=self.prompt_temps_dic[prompt_id]
        if prompt_id in ['004','005']:
            return promt.format(instructions,data['sentence'])
        elif prompt_id=='003':
            return promt.format(instructions,data['context'],data['sentence'])
        elif prompt_id in ['001','002','006', '007']:
            return promt.format(instructions,data['sentence1'],data['sentence2'])
        raise ValueError('Prompt ID not coded')

    def get_data_label(self,data,prompt_id):

        if prompt_id in ['001','002','003','004','005','006','007']:
            return data['label']
                
        raise ValueError('Prompt ID not coded')
        
    def get_input_data_without_def(self,data,prompt_id):          
        
        promt=self.prompt_temps_dic_without_def[prompt_id]
        if prompt_id in ['004','005']:
            return promt.format(data['sentence'])
        elif prompt_id=='003':
            return promt.format(data['context'],data['sentence'])
        elif prompt_id in ['001','002','006','007']:
            return promt.format(data['sentence1'],data['sentence2'])
        raise ValueError('Prompt ID not coded')
        

class Tasks():
    
    source_tasks=['ag_news','ARC-Easy','boolq','commonsense_qa','conll2003_pos','conll2003_ner','mnli','qqp','race','sst2']
    target_tasks=['ARC-Challenge','comve_t1','comve_t2','financial_phrasebank','medmcqa','sciq','social_i_qa','medical-abstracts-tc','scicite']
    
    target_demos={
        'ARC-Challenge':'Question: Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?\nA. "worldwide disease \nB. global mountain building \nC. rise of mammals that preyed upon plants and animals \nD. impact of an asteroid created dust that blocked the sunlight \n Answer: D \n',
        'boolq':'Context: Newcastle upon Tyne (locally /njuːˈkæsəl/ ( listen)), commonly known as Newcastle, is a city in Tyne and Wear, North East England, 103 miles (166 km) south of Edinburgh and 277 miles (446 km) north of London on the northern bank of the River Tyne, 8.5 mi (13.7 km) from the North Sea. Newcastle is the most populous city in the North East, and forms the core of the Tyneside conurbation, the eighth most populous urban area in the United Kingdom. Newcastle is a member of the English Core Cities Group and is a member of the Eurocities network of European cities. Question: is newcastle upon tyne the same as newcastle"? \n Answer: true \n',
        'medmcqa':'Question: Growth hormone has its effect on growth through?\nA. Directly \nB. IG1-1 \nC. Tyroxine \nD. Intranuclear receptors \n Answer: B \n',
        'sciq':'Question: What type of organism is commonly used in preparation of foods such as cheese and yogurt?\nA. Viruses \nB. Protozoa \nC. Gymnosperms \nD. Mesophilic organisms \n Answer: D \n',
        'social_i_qa':'Context: Cameron decided to have a barbecue and gathered her friends together.\n Question: How would Others feel as a result?\nA. Like attending \nB. Like staying home \nC. A good friend to have \n Answer: A \n',
        'financial_phrasebank':'Sentence: For the last quarter of 2010 , Componenta \'s net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .\n Label: positive \n',
        }    
    
    aligner="Aligner: The previous task relates to {} and had to be labeled {}. Use your reasoning ability to learn from the knowledge learned from the previous task to solve the current one which is {} and has labels {}. The new task is:-\n"
    source_info={
        'ag_news':['text classification','sports, business, technology, or world news'],
        'ARC-Easy':['multiple choice question answering','one of the provided options'],
        'race':['read comprehension type multiple choice question answering','one of the provided options'],
        'commonsense_qa':['multiple choice question answering in common-sense reasoning','one of the provided options'],
        'boolq':['question answering','true or false'],
        'conll2003_pos':['sequence classification','into part-of-speech tags'],
        'conll2003_ner':['sequence classification','into name-entity tags'],
        'mnli':['text classification of two sentences','neutral, contradiction or entailment'],
        'qqp':['text classification of two sentences','duplicate or not duplicate'],
        'sst2':['text classification of reviews','negative or positive']
    }
    target_info={
        'ARC-Challenge':['multiple choice question answering','one of the provided options'],
        'comve_t1':['common-sense reasoning','one or two'],
        'comve_t1':['common-sense reasoning','one of the provided options'],
        'boolq':['multiple choice question answering','true or false'],
        'medmcqa':['multiple choice question answering of medical questions','one of the provided options'],
        'sciq':['multiple choice question answering of science questions','one of the provided options'],
        'social_i_qa:':['multiple choice question answering in social common-sense reasoning','one of the provided options'],
        'financial_phrasebank':['text classification of reviews','duplicate or not duplicate'],
        'sst2':['text classification','negative or positive']
    }
    
    def get_aligner(self,source,target):
        return self.aligner.format(self.source_info[source][0],self.source_info[source][1],self.target_info[target][0],self.target_info[target][1])
    
    def get_one_example(self,target_name):
        return self.target_demos[target_name]