import unittest
from pylm.interpolation_model import (
        InterpolationModel, InterpolationModelException)

class TestInterpolationModel(unittest.TestCase):

    def test_interpolation_model(self):
        sentences = [
            ['This', 'is', 'a', 'sentence', '.'],
            ['I', 'like', 'Python', '.']]
        model = InterpolationModel(sentences, 1, [1])
        self.assertEqual(model.prob('I'), 1/9)
        self.assertEqual(model.prob('.'), 2/9)
        self.assertRaises(
            InterpolationModelException,
            InterpolationModel, sentences, 1, [0.5])
        model = InterpolationModel(sentences, 2, [0.75, 0.25])
        self.assertEqual(model.prob('.', ['Python']), 0.75*1 + 0.25*2/9)
        self.assertEqual(model.prob('.', ['like']), 0.75*0 + 0.25*2/9)

if __name__ == '__main__':
    unittest.main()