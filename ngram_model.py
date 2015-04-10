from language_model import LanguageModel
from collections import defaultdict
from itertools import islice, chain

class NgramModel(LanguageModel):
    """Model that only uses highest order ngrams and no smoothing."""

    def __init__(self, sentences, n, pad_left=True, pad_right=False,
            pad_symbol=''):
        """
        Construct NgramModel.

        Params:
            sentences: [iterable of strings] Sequence of sequence of tokens that
                make up a sentence.
                Ex: [['This', 'is', 'a' 'sentence', '.'],
                     ['This', 'is', 'another', 'sentence']]
            n: [int] The highest order model to use. Ex: 3 for a trigram model.
            pad_left: [bool] Whether to pad each sentence by n - 1 pad_symbol
                tokens at the beginning of the sentence.
            pad_right: [bool] Whether to pad each sentence by n - 1 pad_symbol
                tokens at the end of the sentence.
        """
        self.ngram_frequencies = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            if pad_left:
                sentence = chain((pad_symbol,) * (n - 1), sentence)
            if pad_right:
                sentence = chain((pad_symbol,) * (n - 1), sentence)
            context = [t for t in islice(sentence, n - 1)]
            for token in sentence:
                self.ngram_frequencies[tuple(context)][token] += 1
                context.pop(0)
                context.append(token)
        self.n = n

    def order(self):
        """
        Return the count of the highest order model used.
        """
        return self.n

    def prob(self, context, word):
        """
        Get the probability of a word following a context.  i.e. The conditional
        probability P(word|context)

        Param:
            context: [iterable of strings] Sequence of tokens to use as context
                for the word.
            word: [string] Word to find P(word|context) for.
        """
        context = tuple(context)
        if context not in self.ngram_frequencies:
            raise Exception(
                'The context {0} was never seen in training'.format(context))
        pos_words = self.ngram_frequencies[context]
        pos_words_count = sum(pos_words.values())
        return pos_words[word]/pos_words_count
