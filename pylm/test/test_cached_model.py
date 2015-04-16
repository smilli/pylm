import unittest
from pylm.lang_model import InterpolationModel, CachedModel, CachedModelException
from pylm.util import mle_pdist, mle_cpd, ngram_cfd

class TestCachedModel(unittest.TestCase):

    def test_cached_model(self):
        sentences = [
            ['This', 'is', 'a', 'sentence', '.'],
            ['I', 'like', 'Python', '.']]
        ngram_cpd = mle_cpd(ngram_cfd(sentences, 2))
        model = InterpolationModel(ngram_cpd, 2, [0.75, 0.25])
        cache_fdist = {'Python': 2, 'cats': 3}
        cache_pdist = mle_pdist(cache_fdist)
        cached_model = CachedModel(model, cache_pdist, 0.25)
        self.assertEqual(cached_model.prob('Python', ['like']),
                0.75*(0.75*1 + 0.25*1/9) + 0.25*(2/5))
        self.assertEqual(cached_model.prob('.', ['like']),
                0.75*(0.75*0 + 0.25*2/9) + 0.25*0)

if __name__ == '__main__':
    unittest.main()

