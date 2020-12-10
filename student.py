"""
Answer to Question:
My program uses space for simple tokenizer. It removes characters except 'a-zA-Z0-9#@_$%' in the preprocessing.
In the postprocessing stage, if the word occurs less than 3 times when training, 0 will be given. For stopwords,
I initially used the python stopwords package, but without it, the weighted score went up. Hence, I finally selected
some words that are not relevant to sentence meaning. For word dimension, 200 and 300 both perform well, 300 is chosen
in my program.

The net model applies LSTM with CrossEntropy as loss function. LSTM is 2-layer with dropout rate 0.1 to avoid overfitting.
I had tried CNN before, but it performed poorly with weighted score than LSTM. For loss function CrossEntropy, it is
very useful for classification training. Because there are two classification problems of rating and category, I set
the loss to be 0.5*rating_loss plus 0.5*category_loss. For optimizer, initially it is SGD and has a poor performance. I
changed it to Adam, which can adapt to larger data sets and better deal with sparse sets with high computational efficiency.
"""
#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import re
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    processed = sample.split(' ')

    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    illegal_pattern = r'[^a-zA-Z0-9#@_$%\s]'
    sample = [re.sub(illegal_pattern, '', word) for word in sample]

    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    wordFreq = vocab.freqs
    wordItos = vocab.itos
    for b in batch:
        for x, y in enumerate(b):
            if wordFreq[wordItos[y]] < 3:
                b[x] = 0
    return batch


stopWords = {'for', 'a', 'i', 'was', 'it', 'and', 'the', 'are',
             'its', 'his', 'to', 'her', 'him', 'is', 'you',
             'were', 'they', 'my', 'your', 'in', 'am'}
wordVectorDimension = 300
wordVectors = GloVe(name='6B', dim=wordVectorDimension)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ratingOutput = torch.argmax(ratingOutput, dim=1).long()
    categoryOutput = torch.argmax(categoryOutput, dim=1).long()
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """
    # LSTM
    def __init__(self):
        super(network, self).__init__()
        self.input_size = wordVectorDimension
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.lstm = tnn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, dropout=self.dropout)
        self.linear1 = tnn.Linear(self.hidden_size, 2)
        self.linear2 = tnn.Linear(self.hidden_size, 5)

    def forward(self, input, length):
        output, (hidden, cell) = self.lstm(input)
        ratingOutput = self.linear1(output)
        ratingOutput = ratingOutput[:,-1,:]
        categoryOutput = self.linear2(output)
        categoryOutput = categoryOutput[:,-1,:]
        return ratingOutput, categoryOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.cross_entropy_loss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        rate_loss = self.cross_entropy_loss(ratingOutput, ratingTarget)
        category_loss = self.cross_entropy_loss(categoryOutput, categoryTarget)
        loss = 0.5*rate_loss + 0.5*category_loss
        return loss


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01)
