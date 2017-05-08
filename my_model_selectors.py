import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(
                n_components=num_states, covariance_type="diag", n_iter=1000,
                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        def calc_num_para(hmm: GaussianHMM) -> int:
            """ calculate the number parameters in a hmm """
            # propabilities need to sum up to 1
            npara_initial = hmm.n_features - 1
            # Same argument for the weight matrix, the last row is
            # dependent on the rest
            npara_transition = hmm.n_components * (hmm.n_components - 1)
            # for the emission propabilities just of the gaussian hmm, count the
            # number of non zeros in the mean and the weight matrix
            npara_emission = np.count_nonzero(hmm.means_) * np.count_nonzero(hmm.covars_)
            return npara_initial + npara_transition + npara_emission

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = [self.base_model(i)
                  for i in range(self.min_n_components, self.max_n_components+1)]
        try:
            bic = [-2*m.score(self.X, self.lengths) + calc_num_para(m) *np.log(len(self.X))
                   for m in models]
        except (ValueError, AttributeError) as e:
            return models[self.n_constant]

        return models[bic.index(min(bic))]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm 
    topology optimization." Document Analysis and Recognition, 2003. Proceedings. 
    Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def det_dic_error(hmm: GaussianHMM) -> float:
            p_xi = hmm.score(self.X, self.lengths)

            other_words = {i: self.hwords[i] for i in self.hwords if i != self.this_word}

            p_m = 0
            for i in other_words:
                X, lengths = other_words[i]
                p_m += hmm.score(X, lengths)
            p_m *= 1/(len(other_words) - 1)

            return p_xi - p_m


        models = [self.base_model(i)
                  for i in range(self.min_n_components, self.max_n_components+1)]

        try:
            dic_scores = [det_dic_error(m) for m in models]
            return models[dic_scores.index(max(dic_scores))]

        except (ValueError, AttributeError) as e:
            return models[self.n_constant]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(
                n_components=num_states, covariance_type="diag", n_iter=1000,
                random_state=self.random_state, verbose=False)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def det_cv_error(hmm: GaussianHMM) -> float:

            scores = []
            split_method = KFold(random_state=self.random_state)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                scores.append(
                    hmm.fit(X_train, lengths_train).score(X_test, lengths_test))
            return -np.mean(scores)


        models = [self.base_model(i)
                  for i in range(self.min_n_components, self.max_n_components+1)]

        try:
            cv_scores = [det_cv_error(m) for m in models]
            model = models[cv_scores.index(max(cv_scores))]
            return model.fit(self.X, self.lengths)

        except (ValueError, AttributeError) as e:
            return models[self.n_constant]

