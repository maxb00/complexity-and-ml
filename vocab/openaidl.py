# openaidl.py - For getting openai word embeddings
#               into a text file for notebook use.
import os
import openai
import time
from tqdm import tqdm
from openai.embeddings_utils import get_embedding
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, help='gpt model version', defualt='curie', required=False)
parser.add_argument('-d', '--dest', type=str, help='destination folder', default='./gpt', required=False)
args = parser.parse_args()
assert args.version in ['ada', 'curie', 'davinci', 'babbage'], 'Invalid GPT version. Must be ada, curie, davinci, or babbage'

# get openai key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

vocab = []   # list of all our valid words

print('vocab time')
with open(u'./expanded_vocab.txt', 'r') as f:
    for line in f:
        vocab.append(line.strip('\n'))


with open(f'{args.dest}/gpt_{args.version}.txt', 'w') as f:
    counter = 0
    for i in tqdm(range(len(vocab))):
        word = vocab[i]
        em = get_embedding(word, engine=f"text-similarity-{args.version}-001")

        for p in em:
            f.write(str(p) + ' ')
        f.write("\n")

        # this is an excessive amount of caution
        # due to issues had with throttling in Summer '22
        counter += 1
        if counter % 30 == 29:
            time.sleep(120)  
        
print('done')
