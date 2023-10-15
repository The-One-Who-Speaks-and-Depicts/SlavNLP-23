from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, default_data_collator, TrainingArguments, Trainer
import argparse
import pandas as pd
from datasets import load_dataset
import collections
import numpy as np
import math
import torch
from VERNETDatum import VERNetDatum
import json
from tqdm import tqdm

def transform_dataset(path, type, grams):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for s in f.read().split('\n'):
            if s:
                sentences.append(json.loads(s))
    df = pd.DataFrame(columns=['sentence_id', 'words', 'labels'])
    final_arr = []
    iter = 0
    for i in tqdm(range(len(sentences))):
        for j in range(len(sentences[i]["src"].split(' '))):
            try:
                final_arr.append([iter, sentences[i]["src"].split(' ')[j], sentences[i]["src_lab"].split(' ')[j]])
            except: 
                final_arr.append([iter, sentences[i]["src"].split(' ')[j], "c"])
        iter = iter + 1    
    for i in tqdm(range(len(sentences))):
        for j in range(len(sentences[i]["src"].split(' '))):
            if grams == "True":
                for j in range(len(sentences[i]["hyp"])):
                    if "i" in sentences[i]["hyp_lab"][j]:
                        for k in range(len(sentences[i]["hyp"][j].split(' '))):
                            try:
                                final_arr.append([iter, sentences[i]["hyp"][j].split(' ')[k], sentences[i]["hyp_lab"][j].split(' ')[k]])
                            except:    
                                final_arr.append([iter, sentences[i]["hyp"][j].split(' ')[k], "c"])
                        iter = iter + 1
    df = pd.DataFrame(final_arr, columns=['sentence_id', 'words', 'labels'])
    there_is_n_grams = "-with-n-grams" if grams == "True" else ""
    df.to_csv(f'{type}{there_is_n_grams}.csv', index=False)

def main(train_data, dev_data, test_data, grams):
    if train_data:      
        transform_dataset(train_data, "train", grams)
    if dev_data:
        transform_dataset(dev_data, "dev", grams)
    if test_data:
        transform_dataset(test_data, "test", grams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help="Path to train data")
    parser.add_argument('--dev', help="Path to dev data")
    parser.add_argument('--test', help="Path to test data")
    parser.add_argument('--grams', '-f', default="False", help="grams")
    args = parser.parse_args()
    main(args.train, args.dev, args.test, args.grams)