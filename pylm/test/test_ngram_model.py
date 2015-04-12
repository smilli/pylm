import unittest
from pylm.ngram_model import NgramModel

class TestNgramModel(unittest.TestCase):

    def test_ngram_model(self):
        sentences = [
            ['This', 'is', 'a', 'sentence', '.'],
            ['I', 'like', 'Python', '.']]
        model = NgramModel(sentences, 1)
        self.assertEqual(model.prob('I'), 1/9)
        self.assertEqual(model.prob('.'), 2/9)
        self.assertEqual(model.prob('.', ['Python']), 0)
        model = NgramModel(sentences, 2)
        self.assertEqual(model.prob('.', ['Python']), 1)

if __name__ == '__main__':
    unittest.main()
