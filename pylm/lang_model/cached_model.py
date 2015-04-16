from pylm.lang_model import LanguageModel

class CachedModel(LanguageModel):

    def __init__(self, lang_model, cache_lang_model, cache_weight):
        """
        Construct CachedModel.

        params:
            lang_model: [LanguageModel] The baseline language model to use.
            cache_lang_model: [LanguageModel] Langauge model of the cached
                words.
            cache_weight: [float] The amount to weight P(word|cache). The
                lang_model prob will be weighted by (1 - cache_weight).
        """
        if not 0 <= cache_weight <= 1:
            raise CachedModelException('Cache weight must be between 0 and 1.')
        self.lang_model = lang_model
        self.cache_lang_model = cache_lang_model
        self.cache_weight = cache_weight

    def order(self):
        """
        Return the count of the highest order model used.
        """
        return self.lang_model.order()

    def prob(self, word, context):
        """
        Get the probability of a word following a context.  i.e. The conditional
        probability P(word|context)

        Param:
            word: [string] Word to find P(word|context) for.
            context: [iterable of strings] Sequence of tokens to use as context
                for the word.
        """
        return ((1 - self.cache_weight) * self.lang_model.prob(word, context) +
            self.cache_weight * self.cache_lang_model.prob(word, context))

class CachedModelException:
    pass
