# t5 embeds lol
import sentencepiece
from transformers import AutoTokenizer, T5EncoderModel
import torch
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cache', type=str, help='cache location', default=None, required=False)
parser.add_argument('-v', '--version', type=str, help='t5 model size', default='3b', required=False)
parser.add_argument('-d', '--dest', type=str, help='destination folder', default='./opt', required=False)
args = parser.parse_args()
assert args.version in ['3b', '11b', 'large', 'base', 'small'], 'Invalid T5 version. Must be in [3b, 11b, large, base, small]'

def squeeze(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.squeeze(outputs.last_hidden_state, dim=0)
    return np.array(torch.mean(embeddings, dim=0))


def main():
    print("starting")
    if args.cache is not None:
        tokenizer = AutoTokenizer.from_pretrained(f'google/t5-{args.version}', cache_dir=args.cache)
        model = T5EncoderModel.from_pretrained(f'google/t5-{args.version}', cache_dir=args.cache)

    path = u'./expanded_vocab.txt'

    vocab = []
    embeds = []
    print("building vocab...")
    with open(path, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
    print("done")

    print("getting embeds")
    for i in tqdm(range(0, len(vocab))):
        word = vocab[i]
        em = squeeze(word, tokenizer, model)
        embeds.append(em)
        with open(u'./t5/flan_t5_11b.txt', 'a') as f:
            for p in em:
                f.write(str(p) + ' ')
            f.write('\n')
    print("Done.")


if __name__ == "__main__":
    main()
