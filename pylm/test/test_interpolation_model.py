import unittest
from pylm.lang_model import (
        InterpolationModel, InterpolationModelException)
from pylm.util import ngram_cfd, mle_cpd

class TestInterpolationModel(unittest.TestCase):

    def test_interpolation_model(self):
        sentences = [
            ['This', 'is', 'a', 'sentence', '.'],
            ['I', 'like', 'Python', '.']]
        ngram_cpd = mle_cpd(ngram_cfd(sentences, 1))
        model = InterpolationModel(ngram_cpd, 1, [1])
        self.assertEqual(model.prob('I'), 1/9)
        self.assertEqual(model.prob('.'), 2/9)
        self.assertRaises(
            InterpolationModelException,
            InterpolationModel, sentences, 1, [0.5])
        ngram_cpd = mle_cpd(ngram_cfd(sentences, 2))
        model = InterpolationModel(ngram_cpd, 2, [0.75, 0.25])
        self.assertEqual(model.prob('.', ['Python']), 0.75*1 + 0.25*2/9)
        self.assertEqual(model.prob('.', ['like']), 0.75*0 + 0.25*2/9)

if __name__ == '__main__':
    unittest.main()
