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
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
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
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_BIC = float('inf')
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:

                model = self.base_model(n)

                logL = model.score(self.X, self.lengths)

                # N = number of data points
                # d = feature dimensions
                N, d = self.X.shape

                # number of parameters
                # = transition + (mean+var) + initial state
                p = n * (n - 1) + 2 * d * n + (n - 1)

                BIC = -2 * logL + p * np.log(N)

                if BIC < best_BIC:

                    best_BIC = BIC
                    best_model = model

            except:

                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))

                pass

        if self.verbose and best_model is not None:
            print("best model created for {} with {} states".format(self.this_word, best_model.n_components))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):

        # def build_fit_model(word, n):
        #     """Build a model given word and n

        #     Parameters
        #     ----------
        #     word : str
        #     n : int
        #         Number of hidden states

        #     Returns
        #     ----------
        #     model : GaussianHMM
        #     """
        #     X, lengths = self.hwords[word]

        #     return GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
        #                        random_state=self.random_state, verbose=False).fit(X, lengths)

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_DIC = float('-inf')
        best_model = None

        other_words = [word for word in self.hwords.keys() if word != self.this_word]

        assert len(other_words) == len(self.hwords.keys()) - 1, len(other_words)

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:

                model = self.base_model(n)

                log_i = model.score(self.X, self.lengths)
                other_log_i_s = [model.score(*self.hwords[word]) for word in other_words]

                DIC = log_i - np.mean(other_log_i_s)

                # models = [build_fit_model(word, n) for word in words]
                # log_scores = [models[idx].score(*self.hwords[word]) for idx, word in enumerate(words)]

                # i = words.index(self.this_word)

                # DIC = log_scores[i] - 1 / (len(log_scores) - 1) * (np.sum(log_scores) - log_scores[i])

                if DIC > best_DIC:
                    best_DIC = DIC
                    best_model = model

            except:

                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))

                continue

        if self.verbose:
            print("best model created for {} with {} states".format(self.this_word, best_model.n_components))

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_score = float('-inf')
        best_n = None

        for n in range(self.min_n_components, self.max_n_components + 1):

            if len(self.sequences) > 1:
                split_method = KFold(n_splits=min(3, len(self.sequences)))

                log_L = []

                # Collect a score for each fold
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    train_X, train_length = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_length = combine_sequences(cv_test_idx, self.sequences)

                    try:
                        model = GaussianHMM(n_components=n,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(train_X, train_length)

                        logL = model.score(test_X, test_length)
                        log_L.append(logL)

                    except:

                        log_L.append(float('-inf'))

                logL = np.mean(log_L)

            else:
                try:
                    model = self.base_model(n)
                    logL = model.score(self.X, self.lengths)

                except:
                    logL = float("-inf")

            if logL > best_score:

                best_score = logL
                best_n = n

        if self.verbose:
            print("best model created for {} with {} states".format(self.this_word, best_model.n_components))

        return self.base_model(best_n)
