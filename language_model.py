class LanguageModel:
    """Abstract class for a Language Model."""

    def __init__(self):
        """
        Construct LanguageModel.
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
