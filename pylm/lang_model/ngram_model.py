from collections import defaultdict
from itertools import islice, chain
from pylm.lang_model import LanguageModel
from pylm.util import ngram_cfd, mle_cpd

class NgramModel(LanguageModel):
    """Model that only uses highest order ngrams and no smoothing."""

    def __init__(self, ngram_cpd, n):
        """
        Construct NgramModel.

        Params:
            ngram_cpd: [dict] Conditional probability distribution of ngrams.
            n: [int] The highest order model to use. Ex: 3 for a trigram model.
        """
        self.ngram_cpd = ngram_cpd
        self.n = n

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
        return self.ngram_cpd[context][word]
