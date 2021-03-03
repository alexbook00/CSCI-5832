import numpy as np
import string
import csv
from pprint import pprint

# fill array of positive words
pos_words = []
with open('positive-words.txt') as f:
    for line in f:
        pos_words.append(line.split()[0])

# fill array of negative words
neg_words = []
with open('negative-words.txt') as f:
    for line in f:
        neg_words.append(line.split()[0])

# define pronouns array
pronouns = ['i', 'me', 'mine', 'my', 'you', 'your', 'yours', 'we', 'us', 'ours']

# function to extract features from reviews (turn reviews into vectors)
def get_features_from_reviews(file):
    with open(file, encoding="utf8") as f:
        # initialize array of id_features_class vectors
        id_features_class_vectors = []

        for line in f:
            # initialize features vector
            id_features_class = ['ID', 0, 0, 0, 0, 0, 0, 'Class']

            # getting
            if 'Pos' in file:
                id_features_class[-1] = 1
            elif 'Neg' in file:
                id_features_class[-1] = 0
            else:
                id_features_class[-1] = 999

            # check for feature 3
            if 'no' in line.lower():
                id_features_class[3] = 1

            # check for feature 5
            if '!' in line:
                id_features_class[5] = 1

            # split review by spaces
            arr = line.split()

            # get ID
            id_features_class[0] = arr[0]

            # get feature 6 (length minus one due to the ID being the first 'word')
            id_features_class[6] = np.log(len(arr)-1)

            # get remaining features
            for word in arr[1:]:
                # put word to lowercase and remove any punctuation at end of word
                word = word.lower()
                if word[-1] in string.punctuation:
                    word = word[:-1]

                # check for contribution to feature 1
                if word in pos_words:
                    id_features_class[1] += 1

                # check for contribution to feature 2
                if word in neg_words:
                    id_features_class[2] += 1

                # check for contribution to feature 4
                if word in pronouns:
                    id_features_class[4] += 1

            # append features vector to array holding all of them
            id_features_class_vectors.append(id_features_class)

        f.close()

        return id_features_class_vectors

# Turning hotelPosT-train.txt and hotelNegT-train.txt into a list of vectors held in a .csv file

all_features_vectors = get_features_from_reviews('hotelPosT-train.txt') + get_features_from_reviews('hotelNegT-train.txt')


# Turning .csv file back into vectors and splitting the data up into training and development sets.
# Also keeping the whole set together for final testing before evaluating the provided test set.

# output to csv
with open('Book-Alex-assgn2-part1.csv', 'w', newline='') as outfile:
    csvWriter = csv.writer(outfile, delimiter=',')
    csvWriter.writerows(all_features_vectors)

# load data into an array
with open('Book-Alex-assgn2-part1.csv', newline='') as infile:
    data = list(csv.reader(infile))

    infile.close()

# shuffle data array
np.random.shuffle(data)

# split data into training and development
training_data = data[:int(len(data)*.8)]
development_data = data[int(len(data)*.8):]

# split data and development data into (ID, features, label) 3-tuples
ids_features_labels_training = [[x[0], np.array([float(num) for num in x[1:-1]]), int(x[-1])] for x in training_data]
ids_features_labels_development = [[x[0], np.array([float(num) for num in x[1:-1]]), int(x[-1])] for x in development_data]
ids_features_labels_total = [[x[0], np.array([float(num) for num in x[1:-1]]), int(x[-1])] for x in data]

# Defining Stochastic Gradient Descent function, loss function (Cross Entropy),
# and estimated output function (sigmoid function).

def SGD(L, f, x_y):
    # L is the loss function (in this case we use Cross Entropy)
    # f is our estimated output function (use the sigmoid function: 1/(1 + e^-(w dot x + b)) where w is the weights, x is the features, and b is the bias term)
    # x is the set of training inputs (feature vectors)
    # y is the set of training output (labels)

    # initialize weights and bias
    weights = np.zeros(6)
    bias = 1

    learning_rate = .01

    overall_losses = []

    for _ in range(1000):
        np.random.shuffle(x_y)
        # for each training tuple (x, y) (in random order):
        losses = []
        for i in range(len(x_y)):
            features = x_y[i][1]
            y = x_y[i][2] # this is the label/correct score

            # compute our estimated output
            rawscore = np.dot(weights, features) + bias # w dot x plus b
            y_hat = f(rawscore) # this is the computed score

            # compute loss for bookkeeping
            loss = L(y, y_hat)
            losses.append(loss)

            # compute the gradient weights
            gradient = (y_hat - y) * features # this is delta

            # how should we move the set of weights to maximize loss?
            # go the other way instead
            weights = weights - (learning_rate * gradient)

        overall_losses.append(np.sum(losses))

    # return set of weights
    return weights

def cross_entropy(y, y_hat):
    return -(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))

def sigmoid(score):
    return 1/(1 + np.exp(-score))

# Testing the accuracy of the program by averaging the results of ten different sets of learned weights.

def find_accuracy(weights, ids_features_labels):
    num_correct = 0

    for i in range(len(ids_features_labels)):
        features = ids_features_labels[i][1]
        label = ids_features_labels[i][2]

        estimate = sigmoid(np.dot(weights, features) + 1)
        if (estimate > .5 and label == 1) or (estimate < .5 and label == 0):
            num_correct += 1

    return num_correct/len(ids_features_labels)

accuracies = []

for i in range(10):
    print(i)
    weights = SGD(cross_entropy, sigmoid, ids_features_labels_total)
    accuracies.append(find_accuracy(weights, ids_features_labels_total))

print(np.average(np.array(accuracies)))

# Classifying the given test set using a single set of learned weights.

features_vectors_test = get_features_from_reviews('HW2-testset.txt')
ids_features_labels_test = [[x[0], np.array([float(num) for num in x[1:-1]]), int(x[-1])] for x in features_vectors_test]

weights = SGD(cross_entropy, sigmoid, ids_features_labels_total)

for i in range(len(ids_features_labels_test)):
    features = ids_features_labels_test[i][1]

    estimate = sigmoid(np.dot(weights, features) + 1)
    if estimate > .5:
        ids_features_labels_test[i][2] = 'POS'
    else:
        ids_features_labels_test[i][2] = 'NEG'

with open('Book-Alex-assgn2-out.txt', 'w') as f:
    for item in ids_features_labels_test:
        f.write('{} \t {} \n'.format(item[0], item[2]))
