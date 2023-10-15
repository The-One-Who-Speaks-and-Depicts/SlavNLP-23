import argparse
import stanza
import ntpath
from tqdm import tqdm

def main(args):
    stanza.download('Ukrainian')
    nlp = stanza.Pipeline('uk', processors=['tokenize', 'pos', 'lemma', 'depparse'], use_gpu=True, tokenize_pretokenized=True)
    file = open(args.file, 'r', encoding='utf-8')
    name = ntpath.basename(file.name)
    conllu_sents = [i for i in file.read().split('\n\n') if i]
    file.close()
    processed_sents = []
    for sent in tqdm(conllu_sents):
        processed_sent = []
        conllu_strings = sent.split('\n')
        if "# text = " in conllu_strings:
            continue
        for string in conllu_strings:
            if string.startswith('#'):
                processed_sent.append(f"{string}\n")
        token_strings = [s for s in conllu_strings if not s.startswith('#')]
        sent_to_process = [i.split('\t')[1] for i in token_strings if i]
        miscs = [i.split('\t')[9] for i in token_strings if i]
        tagged_sent = nlp([sent_to_process])
        for _, sentence in enumerate(tagged_sent.sentences):
            for _, token in enumerate(sentence.tokens):
                word = token.words[0]
                processed_sent.append(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats}\t{word.head}\t{word.deprel}\t_\t{miscs[word.id - 1]}\n")
        processed_sents.append(''.join(processed_sent))
    processed_sents = '\n'.join(processed_sents)
    with open(f"stanza_{name}", 'w', encoding='utf-8') as tagged:
        tagged.write(processed_sents)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', required=True, help='File to preprocess')
    args = parser.parse_args()
    main(args)