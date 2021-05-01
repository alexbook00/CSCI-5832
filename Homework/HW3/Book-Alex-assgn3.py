import numpy as np
import string
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

def get_sentences_from_file(filename, isTestSet):
    '''
    :param filename: name of file from which sentences are to be read in
    :param isTestSet: keyword arg based on whether or not correct tags should be expected
    :return: returns list of sentences, each of which is a list of 3-tuples of its tokens (position, token, correct tag)
    '''
    with open(filename) as f:
        sentences = []
        tokens = []
        for line in f:
            if line == '\n':
                sentences.append(tokens)
                tokens = []
            else:
                token = line.split()
                if isTestSet:
                    token.append('prediction here')
                tokens.append(token[1:])
    return sentences

def has_punct(token):
    '''
    :param token: token for which punctuation will be checked
    :return: returns bool stating whether or not the given token contains punctuation
    '''
    for char in token:
        if char in string.punctuation:
            return True
    return False

def has_digit(token):
    '''
    :param token: token for which digits will be checked
    :return: returns bool stating whether or not the given token contains any digits
    '''
    for char in token:
        if char.isdigit():
            return True
    return False

def token_to_features(sentence, i):
    '''
    :param sentence: array of tuples of the form (token, correct tag), but correct tag is replaced by
    a non-descript string ('prediction here') if getting features for test set
    '''
    # inspiration for this function can be found at the following URLs:
    # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html
    # http://ceur-ws.org/Vol-1691/paper_10.pdf
    # https://nlp.stanford.edu/pubs/nested-ner.pdf

    token = sentence[i][0]

    features = {
        'bias': 1.0,
        # suffixes
        'token[-4:]': token[-4:] if len(token) > 3 else '', # could also try these as 0 or None
        'token[-3:]': token[-3:] if len(token) > 2 else '',
        'token[-2:]': token[-2:] if len(token) > 1 else '',
        # prefixes
        'token[:-4]': token[:-4] if len(token) > 3 else '',
        'token[:-3]': token[:-3] if len(token) > 2 else '',
        'token[:-2]': token[:-2] if len(token) > 1 else '',
        # token in lowercase
        'token.lower()': token.lower(),
        # if token is in all uppercase
        'token.isupper()': token.isupper(),
        # if token's first letter is capitalized
        'token.istitle()': token.istitle(),
        # if token is a number
        'token.isdigit()': token.isdigit(),
        # length of token
        'token.length()': len(token),
        # if token has punctuation
        'token.has_punct()': has_punct(token),
        # if token has digits
        'token.has_digit()': has_digit(token),
    }
    if i > 0:
        # get previous 3 tokens and tags (if/when applicable)
        # for the training set, this will use the given correct tags
        # for the test set, this will use the predicted tags
        token1before = sentence[i-1][0]
        tag1before = sentence[i-1][1]
        features.update({
            'token1before.lower()': token1before.lower(),
            'token1before.length()': len(token1before),
            'tag1before': tag1before,
        })
        if i > 1:
            token2before = sentence[i-2][0]
            tag2before = sentence[i-2][1]
            features.update({
                'token2before.lower()': token2before.lower(),
                'token2before.length()': len(token2before),
                'tag2before': tag2before,
            })
            if i > 2:
                token3before = sentence[i-3][0]
                tag3before = sentence[i-3][1]
                features.update({
                    'token3before.lower()': token3before.lower(),
                    'token3before.length()': len(token3before),
                    'tag3before': tag3before,
                })
    else:
        features['beginning'] = True

    if i < len(sentence)-1:
        # get next 3 tokens and tags (if/when applicable)
        # not using tags after the current token, as this wouldn't be achievable on the test set
        token1after = sentence[i+1][0]
        features.update({
            'token1after.lower()': token1after.lower(),
            'token1after.length()': len(token1after),
        })
        if i < len(sentence)-2:
            token2after = sentence[i+2][0]
            features.update({
                'token2after.lower()': token2after.lower(),
                'token2after.length()': len(token2after),
            })
            if i < len(sentence)-3:
                token3after = sentence[i+3][0]
                features.update({
                    'token3after.lower()': token3after.lower(),
                    'token3after.length()': len(token3after),
                })
    else:
        features['end'] = True

    return features

def main():
    print('Getting training sentences...')
    # list of list of tuples of the form (token, correct tag)
    sentences = get_sentences_from_file('S21-gene-train.txt', False)

    print('Getting feature dicts...')
    correct_tags = []
    feature_dicts = []
    for i, sentence in enumerate(sentences):
        for j, token in enumerate(sentence):
            correct_tags.append(sentence[j][-1])
            features = token_to_features(sentence, j)
            feature_dicts.append(features)

    print('Fitting DictVectorizer...')
    # uses sklearn's DictVectorizer to turn the feature dictionaries into usable (numerical) feature vectors
    v = DictVectorizer(sparse=True)
    x = v.fit_transform(feature_dicts)

    print('Creating feature vectors...')
    feature_vectors = v.transform(feature_dicts)

    print('Fitting Passive Aggresive Classifier...')
    # explanation of this algorithm found at this URL: https://youtu.be/TJU8NfDdqNQ
    pass_agg = PassiveAggressiveClassifier(verbose=True)
    pass_agg.fit(feature_vectors, correct_tags)

    print('Getting testing sentences...')
    test_sentences = get_sentences_from_file('S21-gene-test.txt', True)

    print('Making predictions on test set...')
    # loop through all test sentences
    for i, sentence in enumerate(test_sentences):
        # loop through all tokens in sentence
        for j, (token, pred) in enumerate(test_sentences[i]):
            # get feature dictionary for current token
            features = token_to_features(test_sentences[i], j)
            # vectorize feature dictionary
            feature_vec = v.transform(features)
            # make prediction using trained model
            curr_prediction = pass_agg.predict(feature_vec)
            # store prediction by replacing 'prediction here' in the current tuple
            test_sentences[i][j] = [token, curr_prediction]

    print('Writing predictions to output file...')
    with open('Book-Alex-assgn3-out.txt', 'w') as f:
        for sentence in test_sentences:
            for i, (token, curr_prediction) in enumerate(sentence):
                f.write('{} \t {} \t {} \n'.format(i+1, token, curr_prediction[0]))
            f.write('\n')

    return

if __name__ == '__main__':
    main()
