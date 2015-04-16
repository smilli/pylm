from collections import defaultdict
from pylm.lang_model import LanguageModel
from pylm.util import ngram_cfd, mle_cpd

class InterpolationModel(LanguageModel):
    """Jelinek-Mercer smoothed n-gram model."""

    def __init__(self, ngram_cpd, n, model_weights):
        """
        Construct InterpolationModel.

        Params:
            ngram_cpd: [dict] Conditional probability distribution of ngrams.
            n: [int] The highest order model to use. Ex: 3 for a trigram model.
            model_weights: [list of ints] The constants to multiply the
                probability of from each ngram model by.  Ordered by highest
                ngram to lowest ngram.  For example [1/2, 1/3, 1/6] to
                discount probabilities from trigrams by 1/2, bigrams by 1/3,
                and unigrams by 1/6.
        """
        if len(model_weights) != n:
            raise InterpolationModelException(
                'You must pass in the same number of weights as the '
                'number of ngram models being used.  I.e. For a trigram model, '
                'pass in three weights.')
        if sum(model_weights) != 1:
            raise InterpolationModelException(
                'Weights must sum to 1 for a proper probability '
                'distribution.')
        self.ngram_cpd = ngram_cpd
        self.n = n
        self.weights = model_weights

    def order(self):
        """
        Return the count of the highest order model used.
        """
        return self.n

    def prob(self, word, context=None):
        """
        Get the probability of a word following a context.  i.e. The conditional
        probability P(word|context)

        Param:
            word: [string] Word to find P(word|context) for.
            context: [iterable of strings] Sequence of tokens to use as context
                for the word.
        """
        if not context:
            context = ()
        else:
            context = tuple(context)
        prob = 0
        for i in range(len(context) + 1):
            prob += self.weights[i] * self.ngram_cpd[context[i:]][word]
        return prob

class InterpolationModelException(Exception):
    pass
