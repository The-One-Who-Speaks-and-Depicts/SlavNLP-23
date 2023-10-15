import argparse
from ua_gec import Corpus
import os
from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm
import glob


def main(args):
    random.seed(args.seed)
    if args.dataset_type == 'UD' and not args.language:
        raise ValueError("Insert a language of the UD dataset")
    if args.split > 1 or args.split < 0:
        raise ValueError("Incorrect split value, should be between 0 and 1")
    conllu_final_sents = []
    if args.dataset_type == 'ua_gec':
        files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(args.data_folder) for f in filenames]
        non_tokenised = [f for f in files if "non-prepared" in f]
        tokenised = [f for f in files if "tokenized" in f]
        sentence_pairs = []
        for text_pair_idx in range(len(tokenised)):
            for sentence_pair in zip(Path(tokenised[text_pair_idx]).read_text().split("\n"), Path(non_tokenised[text_pair_idx]).read_text().split("\n")):
                sentence_pairs.append(sentence_pair)
        sample = random.sample(sentence_pairs, k = int(args.split * len(sentence_pairs)))
        sum = 0
        for i in sample:
            sum = sum + len(i[0].split(' '))
        print(f"Size of sample is {sum} tokens")
        swadesh_dataframe = pd.read_csv(args.swadesh_file,sep='\t')
        for i in tqdm(range(len(sample))):
            type_A_encountered = False
            type_B_encountered = False
            id = f"# sent_id = UA_GEC_{i}\n"
            text = f"# text = {sample[i][1]}\n"
            tokens_original = sample[i][0].split(' ')
            tokens = []
            for j in range(len(tokens_original)):
                swadesh_info = ""
                token = tokens_original[j]
                for _, row in swadesh_dataframe.iterrows():
                    forms = row['Forms'].split(',')
                    for form in forms:
                        if token == form:
                            swadesh_info = f"l-SNP=True|l-SNP-Type=Swadesh|Concept={row['Concept']}|Lemma={row['Lemma']}|Swadesh-Split={row['List part']}"
                            if row['List part'] == 'A':
                                type_A_encountered = True
                            else:
                                type_B_encountered = True
                            break
                if not swadesh_info:
                    swadesh_info = "_"
                tokens.append(f"{j}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{swadesh_info}\n")
            if type_A_encountered and type_B_encountered:
                conllu_final_sents.append((id + text + ''.join(tokens), 'W'))
            if type_A_encountered and not type_B_encountered:
                conllu_final_sents.append((id + text + ''.join(tokens), 'A'))
            if not type_A_encountered and type_B_encountered:
                conllu_final_sents.append((id + text + ''.join(tokens), 'B'))
            if not type_A_encountered and not type_B_encountered:
                conllu_final_sents.append((id + text + ''.join(tokens), 'E'))
    if args.dataset_type == 'UD':
        for filename in glob.glob(os.path.join(args.data_folder, "*.conllu")):
            file = open(filename, mode='r', encoding='utf-8')
            conllu_file = file.read()
            file.close()            
            conllu_sents = [i for i in conllu_file.split('\n\n') if i]
            sample = random.sample(conllu_sents, k = int(args.split * len(conllu_sents)))
            sum = 0
            for i in sample:
                sum = sum + len([j for j in i.split('\n') if j and not j.startswith('#')])
            print(f"Size of sample is {sum} tokens")
            swadesh_dataframe = pd.read_csv(args.swadesh_file,sep=',')
            for sent in tqdm(conllu_sents):
                type_A_encountered = False
                type_B_encountered = False
                conllu_strings = [i for i in sent.split('\n') if i]
                out_strings = []
                for string in conllu_strings:
                    if not string.startswith('#'):
                        split_string = string.split('\t')
                        token = split_string[1]
                        lemma = split_string[2]
                        swadesh_info = ""
                        for _, row in swadesh_dataframe.iterrows():
                            swadesh_item = row[args.language]
                            if lemma == swadesh_item:
                                if row['List part'] == 'A':
                                    type_A_encountered = True
                                else:
                                    type_B_encountered = True
                                if lemma == 'азъ' and token != lemma:
                                    swadesh_info = f"l-SNP=True|l-SNP-Type=Swadesh|Concept=me|Lemma=мъня|Swadesh-Split={row['List part']}"
                                    break
                                if lemma == 'ты' and token != lemma:
                                    swadesh_info = f"l-SNP=True|l-SNP-Type=Swadesh|Concept=you-indirect|Lemma=тебя|Swadesh-Split={row['List part']}"
                                    break
                                swadesh_info = f"l-SNP=True|l-SNP-Type=Swadesh|Concept={row['Concept']}|Lemma={row[args.language]}|Swadesh-Split={row['List part']}"
                                break                                
                        if not swadesh_info:
                            swadesh_info = "_"
                        split_string[9] = f"{swadesh_info}\n"
                        new_string = '\t'.join(split_string)
                        out_strings.append(new_string)
                    else:
                        out_strings.append(f"{string}\n")
                new_sentence = ''.join(out_strings)
                if type_A_encountered and type_B_encountered:
                    conllu_final_sents.append((new_sentence, 'W'))
                if type_A_encountered and not type_B_encountered:
                    conllu_final_sents.append((new_sentence, 'A'))
                if not type_A_encountered and type_B_encountered:
                    conllu_final_sents.append((new_sentence, 'B'))
                if not type_A_encountered and not type_B_encountered:
                    conllu_final_sents.append((new_sentence, 'E'))
    omega_dataset = '\n'.join([sent[0] for sent in conllu_final_sents])
    alpha_dataset = '\n'.join([sent[0] for sent in conllu_final_sents if sent[1] == 'A' or sent[1] == 'E'])
    beta_dataset = '\n'.join([sent[0] for sent in conllu_final_sents if sent[1] == 'B' or sent[1] == 'E'])
    with open(args.output_file + '.alpha.conllu', mode='w', encoding='utf-8') as out:
        out.write(alpha_dataset)
    with open(args.output_file + '.beta.conllu', mode='w', encoding='utf-8') as out:
        out.write(beta_dataset)
    with open(args.output_file + '.omega.conllu', mode='w', encoding='utf-8') as out:
        out.write(omega_dataset)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', '-t', choices=['ua_gec', 'UD'], required=True, help='Type of dataset, options: UD and ua_gec')
    parser.add_argument('--data_folder', '-f', required=True, help='Folder with ua_gec/UD files')
    parser.add_argument('--swadesh_file', '-sw', required=True, help='Path to file with Swadesh data in .csv format')
    parser.add_argument('--output_file', '-o', required=True, help='Path to file with Swadesh-tagged UD data in .conllu format')
    parser.add_argument('--split', '-s', default=1, type=float, help='Part of data  that should be taken, from 0 to 1')
    parser.add_argument('--language', '-l', choices=['ua', 'ru', 'be', 'oes'], help='Language of dataset')
    parser.add_argument('-seed', '-sd', default=1590, type=int, help='random seed')
    args = parser.parse_args()
    main(args)