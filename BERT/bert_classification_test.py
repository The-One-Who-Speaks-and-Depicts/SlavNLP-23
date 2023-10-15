import numpy as np
import pandas as pd
from scipy.special import softmax
import argparse

from simpletransformers.ner import NERModel

from tqdm import tqdm


from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main(args):

    test_df = pd.read_csv(args.test, sep=',')    
    curr_id = -1
    final_array_of_sentences = []
    final_array_of_labels = []
    for _, row in test_df.iterrows():
        if row["words"] not in  ["[CLS]", "[EOS]"]:
            if row["sentence_id"] > curr_id:
                curr_id = curr_id + 1
                final_array_of_sentences.append([])
                final_array_of_sentences[curr_id].append(row["words"])
                final_array_of_labels.append([])
                final_array_of_labels[curr_id].append(row["labels"])
            else:
                final_array_of_sentences[curr_id].append(row["words"])
                final_array_of_labels[curr_id].append(row["labels"])
    for i in zip(final_array_of_sentences, final_array_of_labels):
        if len(i[0]) != len(i[1]):
            print(i)
    final_array_of_sentences = [' '.join(sent).strip() for sent in final_array_of_sentences]
    model = NERModel(
        "bert",
        args.model,
        labels=["c", "i"],
        args={"overwrite_output_dir": True, "num_train_epochs": 1, "reprocess_input_data": False, "evaluate_during_training": False}
    )

    print(final_array_of_sentences[0:25])
    
    predictions, _ = model.predict(final_array_of_sentences)
    y_true = []
    y_pred = []
    for i in tqdm(list(zip(predictions, final_array_of_labels))):
        if len(i[0]) == len(i[1]):
            for j in i[0]:
                for _, v in j.items():
                    y_true.append(v)
                    y_pred.append
            for j in i[1]:
                y_pred.append(j)
    print(precision_recall_fscore_support(y_true, y_pred, labels=["c", "i"]))
    print(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=["c", "i"])
    disp.plot()
    plt.savefig('cm.png')
    y_true = [1 if y == "i" else 0 for y in y_true]
    y_pred = [1 if y == "i" else 0 for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'Binarized confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t')
    parser.add_argument('--model', '-m')
    args = parser.parse_args()
    main(args)
