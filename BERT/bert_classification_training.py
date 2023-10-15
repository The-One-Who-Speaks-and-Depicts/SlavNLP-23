import numpy as np
import pandas as pd
from scipy.special import softmax
import argparse
import os
import re
from tqdm import tqdm

from simpletransformers.ner import NERModel

def get_csv_from_conllu_dataset(path, augmentation):
    file = open(path, 'r', encoding='utf-8')
    conllu_file = file.read()
    file.close()
    if not os.path.exists(os.path.join(os.getcwd(), "bert-training-data")):
        print("Creating directory for data...")
        os.makedirs(os.path.join(os.getcwd(), "bert-training-data"))
    sentences = [sent for sent in conllu_file.split("\n\n") if sent]
    final_sents = []
    for sent in tqdm(sentences):
        conllu_lines = [i for i in sent.split('\n') if i]
        id = ""
        for line in conllu_lines:
            if "sent_id" in line:
                id = re.sub('# sent_id = ', "", line)
                id = id.strip()
                break
        final_sents.append([id, 'CLS', 'c'])
        for line in conllu_lines:
            if not line.startswith("#"):
                split_token = line.split('\t')
                if 'l-SNP-Type=Swadesh' in split_token[9]:
                    final_sents.append([id, split_token[1], 'i'])
                else:
                    final_sents.append([id, split_token[1], 'c'])
        final_sents.append([id, 'EOS', 'c'])
    if augmentation and augmentation == 'n-grams':
        augmented_sents = []
        for i in tqdm(range(len(final_sents))):
            if final_sents[i][2] == 'i':
                for j in [[id, 'CLS', 'c'], final_sents[i - 1], final_sents[i], final_sents[i + 1], [id, 'EOS', 'c']]:
                    augmented_sents.append(j)
        for i in tqdm(augmented_sents):
            final_sents.append(i)
    csv_df = pd.DataFrame(final_sents, columns=['sentence_id', 'words', 'labels'])
    name = "train.csv" if not augmentation else "train-with-n-grams.csv"
    csv_df.to_csv(os.path.join(os.getcwd(), "bert-training-data", name), index=False)


def main(args):

    #get_csv_from_conllu_dataset(args.train, args.augmentation)

    # name = "train.csv" if not args.augmentation else "train-with-n-grams.csv"

    # train_df = pd.read_csv(os.path.join(os.getcwd(), "bert-training-data", name), sep=',')
    train_df = pd.read_csv(args.train, sep=',')

    # Create a NERModel
    model = NERModel(
        "bert",
        "bert-base-multilingual-uncased",
        labels=["c", "i"],
        args={"overwrite_output_dir": False, "num_train_epochs": 1, "train_batch_size": 1, "reprocess_input_data": True, "evaluate_during_training": False,}
    )
    print(model.args.train_batch_size)

    # # Train the model
    model.train_model(train_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t')
    parser.add_argument('--augmentation', '-a', choices=['n-grams'], default=None)
    args = parser.parse_args()
    main(args)

