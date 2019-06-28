from sklearn.utils.class_weight import compute_class_weight
import os
import pickle
import re

import numpy as np
from keras import Sequential
from keras.layers import GRU, LSTM, Dense, Embedding, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer


def load_data(path):

    sentences = []
    labels = []
    targets = []

    with open(path) as file:
        lines = file.readlines()

    for l in lines:
        sentences.append(re.search(' (.*) EOS', l).group(1))
        labels.append(re.search('\t(.*) atis', l).group(1).split())
        targets.append(re.search('atis_(.*)', l).group(0))

    return np.array(sentences), np.array(labels), np.array(targets)


def preprocess(sentences, targets, tokenizer=None, summary=False, labelizer=None, statssummary=False, pad=False, replace_dig=True):

    stats = {}

    vocab = set()
    for s in sentences:
        if replace_dig:
            s = re.sub(r'\d', 'DIG', s)
        vocab |= set(s.split())

    lengths = [len(s.split()) for s in sentences]

    stats['nb_sentences'] = len(sentences)
    stats['nb_words'] = len(vocab)
    stats['nb_classes'] = len(np.unique(targets))
    stats['class_cnt'] = dict(zip(*np.unique(targets, return_counts=True)))
    stats['max_len'] = max(lengths)
    stats['weights'] = compute_class_weight(
        'balanced', np.unique(targets), targets)

    if summary:
        print(f'Total: {len(sentences)} sentences ')
        print(f'Max sentence length {max(lengths)}')
        print(f'Number of words: {len(vocab)}')
        print(f'Number of classes: {len(np.unique(targets))}')

    if tokenizer is None:
        tokenizer = Tokenizer(stats['nb_words'], oov_token='UNK')
        tokenizer.fit_on_texts(sentences)

        if not os.path.isdir('./out/'):
            os.mkdir('./out/')

        with open('./out/tokenizer.pickle', 'wb') as file:
            pickle.dump(tokenizer, file)

    X = tokenizer.texts_to_sequences(sentences)

    if pad:
        X = pad_sequences(X, maxlen=stats['max_len'])
    else:
        X = np.asarray([np.asarray(xx) for xx in X])

    if labelizer is None:
        labelizer = LabelBinarizer()

        labelizer.fit(targets)

        with open('./out/label_encoder.pickle', 'wb') as file:
            pickle.dump(labelizer, file)

    y = labelizer.transform(targets)

    return X, y, stats


