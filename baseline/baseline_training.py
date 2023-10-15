import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


def main(args):
    file = open(args.train_file, 'r', encoding='utf-8')
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
    y = []
    for line in conllu:
        if line and line.strip() and not line.startswith('#'):
            split_token = line.split('\t')
            X.append([lemmata[split_token[2]], lemmata[split_token[2]] / total_tokens])
            if "l-SNP-Type=Swadesh" in split_token[9]:
                y.append(1)
            else:
                y.append(0)
    X = np.array(X)
    y = np.array(y)
    clf = RandomForestClassifier(random_state=1590).fit(X, y)
    pickle.dump(clf, open(args.model_name + '.model', 'wb')) 
    

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', '-t', required=True)
    parser.add_argument('--model_name', '-m', required=True)
    args = parser.parse_args()
    main(args)