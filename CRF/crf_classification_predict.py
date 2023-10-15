import argparse
import pandas as pd       
import nltk
import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re
from tqdm import tqdm
import json
import pickle
import os

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from itertools import cycle

def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word': word
    }    
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word': word1,
        })     
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word': word1,
        })

    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def format_data(path):
    file = open(path, 'r', encoding='utf-8')
    conllu = file.read()
    sents = [i for i in conllu.split('\n\n') if i and i.strip()]
    file.close()
    sentences = []
    for sent in sents:
        words = ['[CLS]']
        for line in sent.split('\n'):    
            if line and line.strip() and not line.startswith('#'):
                split_token = line.split('\t')
                words.append(split_token[1])
        words.append('[EOS]')
        sentences.append(words)
    return sentences

def main(args):
    test_sents = format_data(args.predict)
    X_test = [sent2features(s) for s in test_sents]
    crf = pickle.load(open(args.path, 'rb'))
    y_pred = crf.predict(X_test)
    final_lines = []
    file = open(args.predict, 'r', encoding='utf-8')
    conllu = file.readlines()
    for i in zip(y_pred, conllu):
        preds = i[0][1:len(i[0]) - 1]
        if len(preds) == 0:
            for line in i[1].split('\n'):
                final_lines.append(f"{line}\n")
                final_lines.append("\n")
                continue  
        preds_cycle = cycle(preds)
        for line in i[1].split('\n'):
            if line and line.strip():
                if line.startswith('#'):
                    final_lines.append(f"{line}\n")
                else:
                    pred = next(preds_cycle)
                    swadesh_info = '1' if pred == '1' else '0'
                    split_token = line.split('\t')
                    final_lines.append(f"{split_token[0]}\t{split_token[1]}\t{split_token[2]}\t{split_token[3]}\t{split_token[4]}\t{split_token[5]}\t{split_token[6]}\t{split_token[7]}\t{split_token[8]}\tl-SNP-Prediction={swadesh_info}\t")          
        final_lines.append('\n')
    predicted_dataset = ''.join(final_lines)
    d = os.path.join(os.path.split(args.experiment_name)[0])
    if not os.path.exists(d):
        os.makedirs(d)
    with open(os.path.join(d, 'test_out.conllu'), 'w', encoding='utf-8') as f:
        f.write(predicted_dataset)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', '-pr')
    parser.add_argument('--path', '-p')
    parser.add_argument('--experiment-name', '-e')
    args = parser.parse_args()
    main(args)
