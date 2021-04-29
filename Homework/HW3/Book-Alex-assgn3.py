import numpy as np
import string
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

le = {}

words = []

def get_sentences_from_file(filename):
    with open(filename) as f:
        unique_words = set()
        sentences = []
        tokens = []
        for line in f:
            # if empty line (new sentence) clear the tokens array
            if line == '\n':
                sentences.append(tokens)
                tokens = []
                # break
            else:
                token = line.split()
                tokens.append(token)
                unique_words.add(token[1].lower())
    return sentences, unique_words

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
    global le
    feature_dict_list = []

    for i, (position, word, tag) in enumerate(sentence):

        # ideas for feature dictionary taken from the following URLs:
        # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#training
        # http://ceur-ws.org/Vol-1691/paper_10.pdf
        # https://nlp.stanford.edu/pubs/nested-ner.pdf
        position = int(position)
        features = {
            'bias': 1.0,
            # suffixes
            'word[-4:]': word[-4:] if len(word) > 3 else '', # could also try these as 0 or None
            'word[-3:]': word[-3:] if len(word) > 2 else '',
            'word[-2:]': word[-2:] if len(word) > 1 else '',
            # prefixes
            'word[:-4]': word[:-4] if len(word) > 3 else '',
            'word[:-3]': word[:-3] if len(word) > 2 else '',
            'word[:-2]': word[:-2] if len(word) > 1 else '',
            # 'word.lower()': le[word.lower()] if word.lower() in le.keys() else 0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.length()': len(word),
            'word.has_punct()': has_punct(word),
            'word.has_digit()': has_digit(word),
        }
        if position > 1:
            # get previous 3 words and tags (if/when applicable)
            word1before = sentence[i-1][1]
            tag1before = sentence[i-1][2]
            features.update({
                # 'word1before.lower()': le[word1before.lower()] if word1before.lower() in le.keys() else 0,
                'word1before.lower()': word1before.lower(),
                'word1before.length()': len(word1before),
                'tag1before': tag1before,
            })
            if position > 2:
                word2before = sentence[i-2][1]
                tag2before = sentence[i-2][2]
                features.update({
                    # 'word2before.lower()': le[word2before.lower()] if word2before.lower() in le.keys() else 0,
                    'word2before.lower()': word2before.lower(),
                    'word2before.length()': len(word2before),
                    'tag2before': tag2before,
                })
                if position > 3:
                    word3before = sentence[i-3][1]
                    tag3before = sentence[i-3][2]
                    features.update({
                        # 'word3before.lower()': le[word3before.lower()] if word3before.lower() in le.keys() else 0,
                        'word3before.lower()': word3before.lower(),
                        'word3before.length()': len(word3before),
                        'tag3before': tag3before,
                    })
        else:
            features['beginning'] = True

        if position < len(sentence):
            # get next 3 words and tags (if/when applicable)
            word1after = sentence[i+1][1]
            tag1after = sentence[i+1][2]
            features.update({
                # 'word1after.lower()': le[word1after.lower()] if word1after.lower() in le.keys() else 0,
                'word1after.lower()': word1after.lower(),
                'word1after.length()': len(word1after),
            })
            if position < len(sentence)-1:
                word2after = sentence[i+2][1]
                tag2after = sentence[i+2][2]
                features.update({
                    # 'word2after.lower()': le[word2after.lower()] if word2after.lower() in le.keys() else 0,
                    'word2after.lower()': word2after.lower(),
                    'word2after.length()': len(word2after),
                })
                if position < len(sentence)-2:
                    word3after = sentence[i+3][1]
                    tag3after = sentence[i+3][2]
                    features.update({
                        # 'word3after.lower()': le[word3after.lower()] if word3after.lower() in le.keys() else 0,
                        'word3after.lower()': word3after.lower(),
                        'word3after.length()': len(word3after),
                    })
        else:
            features['end'] = True

        feature_dict_list.append(features)

    return feature_dict_list

def main():
    global le
    print('Creating feature dictionaries...')
    sentences, unique_words = get_sentences_from_file('S21-gene-train.txt')

    le = {word:i for i,word in enumerate(unique_words, 1)}

    print('Getting feature dicts...')
    correct_tags = [] # list of all 3-tuples
    feature_dicts = []
    for sentence in sentences:
        feature_dicts.extend(sentence_to_feature_dicts(sentence))
        for token in sentence:
            correct_tags.append(token[-1])


    print('Fitting DictVectorizer...')
    v = DictVectorizer(sparse=True)
    x = v.fit_transform(feature_dicts)

    import os, psutil
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)

    print('Creating feature vectors...')
    # feature_vectors = []
    feature_vectors = v.transform(feature_dicts)
    print(feature_vectors.shape)
    # for f_dict in feature_dicts:
    #     # f_vect = np.matrix(v.transform(f_dict).toarray())
    #     feature_vectors.append(v.transform(f_dict))

    print('Fitting SGDClassifier...')
    clf = PassiveAggressiveClassifier(verbose=True)
    clf.fit(feature_vectors, correct_tags)

    return

if __name__ == '__main__':
    main()
