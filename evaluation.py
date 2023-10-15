import argparse

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main(args):
    gold = open(args.gold, 'r', encoding='utf-8')
    gold_lines = [l for l in gold.readlines() if l and l.strip() and not l.startswith('#')]
    tagged = open(args.tagged, 'r', encoding='utf-8')
    tagged_lines = [l for l in tagged.readlines() if l and l.strip() and not l.startswith('#')]
    y = {}
    y_true = []
    y_pred = []
    for i in zip(gold_lines, tagged_lines):
        split_token = i[0].split('\t')
        misc = split_token[9]
        pred = 0 if "l-SNP-Prediction=0" in i[1] else 1
        y_pred.append(pred)
        if "l-SNP-Type=Swadesh" in misc:
            y_true.append(1)
        else:
            y_true.append(0)    
    print(precision_recall_fscore_support(y_true, y_pred, labels=["0", "1"]))
    print(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=["0", "1"])
    disp.plot()
    plt.savefig('cm.png')
    y_true = [1 if y == "1" else 0 for y in y_true]
    y_pred = [1 if y == "1" else 0 for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'Binarized confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', '-g', required=True, help='Original file with correct labels')
    parser.add_argument('--tagged', '-t', required=True, help='New file, tagged by a model of choice')
    args = parser.parse_args()
    main(args)
