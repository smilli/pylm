import unittest
from pylm.util import ngrams, ngram_cfd

class TestNgrams(unittest.TestCase):

    def test_ngrams(self):
        sentence = ['My', 'favorite', 'anime', 'is', 'Code', 'Geass', '.']
        sent_ngrams = list(ngrams(sentence, 3))
        for s in ngrams(sentence, 3):
            print(s)
        self.assertEqual(len(sent_ngrams), len(sentence))
        self.assertEqual(sent_ngrams,
            [('', '', 'My'), ('', 'My', 'favorite'),
            ('My', 'favorite', 'anime'), ('favorite', 'anime', 'is'),
            ('anime', 'is', 'Code'), ('is', 'Code', 'Geass'),
            ('Code', 'Geass', '.')])

    def test_ngram_cfd(self):
        sentences = [
            ['This', 'is', 'a', 'sentence', '.'],
            ['I', 'like', 'Python', '.']]
        cfd = ngram_cfd(sentences, 1)
        unique_words = set(w for s in sentences for w in s)
        self.assertEqual(len(cfd.keys()), 1)
        self.assertEqual(len(cfd[()].keys()), len(unique_words))
        self.assertEqual(cfd[()]['.'], 2)
        cfd = ngram_cfd(sentences, 2)
        self.assertEqual(len(cfd.keys()), 9)
        self.assertEqual(len(cfd[('',)].keys()), 2)
        cfd = ngram_cfd(sentences, 3)
        self.assertEqual(len(cfd.keys()), 17)

if __name__ == '__main__':
    unittest.main()
