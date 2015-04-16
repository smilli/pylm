import unittest
from pylm.lang_model import NgramModel
from pylm.util import ngram_cfd, mle_cpd

class TestNgramModel(unittest.TestCase):

    def test_ngram_model(self):
        sentences = [
            ['This', 'is', 'a', 'sentence', '.'],
            ['I', 'like', 'Python', '.']]
        ngram_cpd = mle_cpd(ngram_cfd(sentences, 1))
        model = NgramModel(ngram_cpd, 1)
        self.assertEqual(model.prob('I'), 1/9)
        self.assertEqual(model.prob('.'), 2/9)
        self.assertEqual(model.prob('.', ['Python']), 2/9)
        ngram_cpd = mle_cpd(ngram_cfd(sentences, 2))
        model = NgramModel(ngram_cpd, 2)
        self.assertEqual(model.prob('.', ['Python']), 1)
        self.assertEqual(model.prob('.', ['like', 'Python']), 1)

if __name__ == '__main__':
    unittest.main()
