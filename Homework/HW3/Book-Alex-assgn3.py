import numpy as np
import string
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer

words = []

with open('S21-gene-train.txt') as f:
    sentences = []
    for line in f:
        tokens = []
        # if empty line (new sentence) clear the tokens_in_sentence array
        if line == '\n':
            sentences.append(tokens)
            tokens = []
        else:
            tokens.append(line.split())

def has_punct(word):
    for char in word:
        if char in string.punctuation:
            return True
    return False

def has_digit(word):
    for char in word:
        if char.isdigit():
            return True
    return False

def sentence_to_feature_dicts(sentence):
    '''
    :param sentence: list of 3-tuples of the form (position in sentence, word, correct IOB tag)
    :return: returns feature dictionary
    '''
    feature_dict_list = []

    for i, (position, word, tag) in enumerate(sentence):
        # position, word, tag = token

        # ideas for feature dictionary taken from the following URLs:
        # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#training
        # http://ceur-ws.org/Vol-1691/paper_10.pdf
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            # suffixes
            'word[-4:]': word[-4:] if len(word) > 3 else '', # could also try these as 0 or None
            'word[-3:]': word[-3:] if len(word) > 2 else '',
            'word[-2:]': word[-2:] if len(word) > 1 else '',
            # prefixes
            'word[:-4]': word[:-4] if len(word) > 3 else '',
            'word[:-3]': word[:-3] if len(word) > 2 else '',
            'word[:-2]': word[:-2] if len(word) > 1 else '',
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.length()': word.length(),
            'word.has_punct()': word.has_punct(),
            'word.has_digit()': word.has_digit(),
        }
        if position > 1:
            # get previous 3 words and tags (if/when applicable)
            pass
        else:
            features['beginning'] = True
        if i < len(sentence):
            # get next 3 words and tags (if/when applicable)
            pass
        else:
            features['end'] = True

        feature_dict_list.append(features)

    return feature_dict_list
