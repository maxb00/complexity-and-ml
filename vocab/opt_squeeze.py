# opt_squeeze.py - A script for saving our common vocab as OPT-13b embeds
from transformers import GPT2Tokenizer, OPTModel
import torch
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cache', type=str, help='cache location', default=None, required=False)
parser.add_argument('-v', '--version', type=str, help='OPT model size', default='13b', required=False)
parser.add_argument('-d', '--dest', type=str, help='destination folder', default='./opt', required=False)
args = parser.parse_args()
assert args.version in ['13b', '6.7b', '2.7b', '1.3b', '350m', '125m'], 'Invalid OPT version. Must be in [13b, 6.7b, 2.7b, 1.3b, 350m, 125m]'


def squeeze(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.squeeze(outputs.last_hidden_state, dim=0)
    return np.array(torch.mean(embeddings[1:], dim=0))


def main():
    print("starting")
    if args.cache is not None:
        tokenizer = GPT2Tokenizer.from_pretrained(
            f"facebook/opt-{args.version}", cache_dir=args.cache)
        model = OPTModel.from_pretrained(
            f"facebook/opt-{args.version}", cache_dir=args.cache)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(f"facebook/opt-{args.version}")
        model = OPTModel.from_pretrained(f"facebook/opt-{args.version}")

    path = u'./expanded_vocab.txt'

    vocab = []
    embeds = []
    print("building vocab...")
    with open(path, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
    print("done")

    print("getting embeds")
    for i in tqdm(range(3456, len(vocab))):
        word = vocab[i]
        em = squeeze(word, tokenizer, model)
        embeds.append(em)
        with open(f'{args.dest}/{args.version}.txt', 'a') as f:
            for p in em:
                f.write(str(p) + ' ')
            f.write('\n')
    print("Done.")


if __name__ == "__main__":
    main()
