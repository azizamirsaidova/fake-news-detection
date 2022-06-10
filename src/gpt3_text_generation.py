import pandas as pd
from transformers import pipeline

uri_train  = 'https://raw.githubusercontent.com/azizamirsaidova/fake-news-detection/main/liar-dataset/train.tsv'
uri_valid  = 'https://raw.githubusercontent.com/azizamirsaidova/fake-news-detection/main/liar-dataset/valid.tsv'
uri_test  = 'https://raw.githubusercontent.com/azizamirsaidova/fake-news-detection/main/liar-dataset/test.tsv'

    
df_train = pd.read_table(uri_train,
                             names = ['id',	'label'	,'statement',	'subject',	'speaker', 	'job', 	'state',	'party',	'barely_true_c',	'false_c',	'half_true_c',	'mostly_true_c',	'pants_on_fire_c',	'venue'])

    
df_valid = pd.read_table(uri_valid,
                             names =['id',	'label'	,'statement',	'subject',	'speaker', 	'job', 	'state',	'party',	'barely_true_c',	'false_c',	'half_true_c',	'mostly_true_c',	'pants_on_fire_c',	'venue'])


df_test = pd.read_csv(uri_test, sep='\t', 
                            names =['id',	'label'	,'statement',	'subject',	'speaker', 	'job', 	'state',	'party',	'barely_true_c',	'false_c',	'half_true_c',	'mostly_true_c',	'pants_on_fire_c',	'venue']) 


df = pd.concat([df_train, df_valid])

dff = df[['label', 'statement']]

false = dff[['statement']].loc[df['label'] == 'false']
true = dff[['statement']].loc[df['label'] == 'true']
false['statement']= false['statement'].astype(str)
true['statement'] = true['statement'].astype(str)
true_50_subset = true['statement'].sample(frac=0.5)

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
all_text_results = generator((list(true_50_subset)), max_length=200, do_sample=True, temperature=0.9, top_k = 50)
all_text_results.to_csv('/Users/azizamirsaidova/Documents/GitHub/fake-news-detection/output/True_Text_Generated.csv')


