import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import os
from itertools import cycle


def main(args):
    d = os.path.split(args.experiment_name)[0]
    if not os.path.exists(d):
        os.makedirs(d)
    clf = pickle.load(open(f"{args.model_name}.model", 'rb'))
    file = open(args.test_file, 'r', encoding='utf-8')
    conllu = file.readlines()
    file.close()
    lemmata = {}
    total_tokens = 0
    for line in conllu:
        if line and line.strip() and not line.startswith('#'):
            total_tokens = total_tokens + 1
            lemma = line.split('\t')[2]
            if lemma in lemmata.keys():
                lemmata[lemma] = lemmata[lemma] + 1
            else:
                lemmata[lemma] = 1
    X = []
    for line in conllu:
        if line and line.strip() and not line.startswith('#'):
            split_token = line.split('\t')
            X.append([lemmata[split_token[2]], lemmata[split_token[2]] / total_tokens])
    X = np.array(X)
    predictions = clf.predict(X)
    predictions = cycle(predictions)
    predicted_dataset = []
    for line in conllu:
        if line and line.strip() and not line.startswith('#'):
            split_token = line.split('\t')
            pred = next(predictions)
            predicted_dataset.append(f"{split_token[0]}\t{split_token[1]}\t{split_token[2]}\t{split_token[3]}\t{split_token[4]}\t{split_token[5]}\t{split_token[6]}\t{split_token[7]}\t{split_token[8]}\tl-SNP-Prediction={pred}\t\n")
        else:
            predicted_dataset.append(line)
    predicted_dataset = ''.join(predicted_dataset)
    with open(os.path.join(d, 'test_out.conllu'), 'w', encoding='utf-8') as f:
        f.write(predicted_dataset)
    

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', '-t', required=True)
    parser.add_argument('--model-name', '-m', required=True)
    parser.add_argument('--experiment-name', '-e', required=True)
    args = parser.parse_args()
    main(args)