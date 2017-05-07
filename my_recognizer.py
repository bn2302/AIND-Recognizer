import warnings
import numpy as np
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for s in test_set.get_all_Xlengths():

        scores = {}
        for m in models:
            try:
                X, lengths = test_set.get_all_Xlengths()[s]
                scores[m] = models[m].score(X, lengths)
            except ValueError as e:
                print('{} : when training model {}, setting loglik to -inf'.format(
                    e, m))
                scores[m] = -np.inf

        probabilities.append(scores)
        key, _ = max(scores.items(), key=lambda x: x[1])
        guesses.append(key)

    return probabilities, guesses
