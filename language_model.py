class LanguageModel:
    """Abstract class for a Language Model."""

    def __init__(self, sentences, n, pad_left=True, pad_right=False,
            pad_symbol=''):
        """
        Construct LanguageModel.

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
        raise NotImplementedError()

    def order(self):
        """
        Return the count of the highest order model used.
        """
        raise NotImplementedError()

    def prob(self, context, word):
        """
        Get the probability of a word following a context.  i.e. The conditional
        probability P(word|context)

        Param:
            context: [iterable of strings] Sequence of tokens to use as context
                for the word.
            word: [string] Word to find P(word|context) for.
        """
        raise NotImplementedError()
