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

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word
    }    
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
        })     
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
        })

    else:
        features['EOS'] = True 
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]

def format_data(path, grams):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for s in f.read().split('\n'):
            if s:
                sentences.append(json.loads(s))
    final_arr = []
    iter = 0
    for i in tqdm(range(len(sentences))):
        final_arr.append([])
        for j in range(len(sentences[i]["src"].split(' '))):
            try:
                final_arr[iter].append([sentences[i]["src"].split(' ')[j], sentences[i]["src_lab"].split(' ')[j]])
            except: 
                final_arr[iter].append([sentences[i]["src"].split(' ')[j], "c"])
        iter = iter + 1    
    for i in tqdm(range(len(sentences))):
        for l in range(len(sentences[i]["src"].split(' '))):
            if grams == "True":
                for j in range(len(sentences[i]["hyp"])):
                    final_arr.append([])
                    if "i" in sentences[i]["hyp_lab"][j]:
                        for k in range(len(sentences[i]["hyp"][j].split(' '))):
                            try:
                                final_arr[iter].append([sentences[i]["hyp"][j].split(' ')[k], sentences[i]["hyp_lab"][j].split(' ')[k]])
                            except:    
                                final_arr[iter].append([sentences[i]["hyp"][j].split(' ')[k], "c"])
                        iter = iter + 1
    return final_arr

def format_conllu_data(path, grams):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for s in f.read().split('\n\n'):
            if s:
                sentences.append(s)
    final_arr = []
    iter = 0
    for i in tqdm(range(len(sentences))):
        final_arr.append([])
        final_arr[iter].append(['[CLS]', '0'])
        for line in sentences[i].split('\n'):
            if line and line.strip() and not line.startswith('#'):
                split_token = line.split('\t')
                final_arr[iter].append([split_token[1], '1' if "l-SNP-Type=Swadesh" in split_token[9] else '0'])
        iter = iter + 1
    if grams == "True":
        for i in tqdm(range(len(sentences))):
            words = sentences[i].split('\n')
            for j in range(len(words)):
                line = words[j]
                if line and line.strip() and not line.startswith('#'):
                    split_token = line.split('\t')
                    if "l-SNP-Type=Swadesh" in split_token[9]:
                        preceding = ['[CLS]', '0'] if split_token[0] == '1' else [words[j - 1].split('\t')[1], '1' if "l-SNP-Type=Swadesh" in words[j - 1].split('\t')[9] else '0']
                        following = ['[EOS]', '0'] if (j == len(words) - 1) else [words[j + 1].split('\t')[1], '1' if "l-SNP-Type=Swadesh" in words[j + 1].split('\t')[9] else '0']
                        final_arr.append([preceding, [split_token[1], '1' if "l-SNP-Type=Swadesh" in split_token[9] else '0'], following])
                    iter = iter + 1
    return final_arr

def main(args):
    final_sents = format_conllu_data(args.train, args.grams)
    Xtrain = [sent2features(s) for s in final_sents]
    ytrain = [sent2labels(s) for s in final_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm = 'lbfgs',
        c1 = 0.25,
        c2 = 0.3,
        max_iterations = 100,
        all_possible_transitions=True
    )
    crf.fit(Xtrain, ytrain)
    pickle.dump(crf, open(args.model_name + '.model', 'wb')) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t')
    parser.add_argument('--model_name', '-m')
    parser.add_argument('--grams', '-g', default="False")
    args = parser.parse_args()
    main(args)
