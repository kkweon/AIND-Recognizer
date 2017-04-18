import warnings
import numpy as np

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """Recognize test word sequences from word models set

    Parameters
    ----------
    models : dict
        Dictionary of Trained models
        {"SOMEWORD": GaussianHMM, "SOMEOTHERWORD": ...}
    test_set : SinglesData

    Returns
    ----------
    probabilities : list
        Probability
        probabilities is a list of dictionaries where each key a word and value is Log Liklihood
            [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             ]

    guesses : list
        Guess
        guesses is a list of the best guess words ordered by the test set word_id
            ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]

        both lists are ordered by the test set word_id
    """
    def get_score(model, X_batch, length_batch):
        """Get model.score(X, length) """
        try:
            score = model.score(X_batch, length_batch)

        except:
            score = float("-inf")

        return score

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    Xlenghts_dict = test_set.get_all_Xlengths()

    for idx in range(test_set.num_items):
        X_batch, length_batch = Xlenghts_dict[idx]

        prob = [(trained_word, get_score(model, X_batch, length_batch)) for trained_word, model in models.items()]

        argmax = np.argmax([v for k, v in prob])
        guess = prob[argmax][0]

        probabilities.append(dict(prob))
        guesses.append(guess)

    return probabilities, guesses
