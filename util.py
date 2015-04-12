from collections import defaultdict

def ngram_cfd(sentences, n):
    """
    Compute n-grams for n ranging from 1 to highest_n.

    Params:
        sentences: [iterable of strings] Sequence of sequence of tokens that
            make up a sentence.
            Ex: [['This', 'is', 'a' 'sentence', '.'],
                    ['This', 'is', 'another', 'sentence', '.']]
        highest_n: [int] Highest order n-grams to compute.

    Returns:
        frequencies [defaultdict] A conditional frequency distribution of the
            counts of a word after the given context.
    """
    frequencies = defaultdict(lambda: defaultdict(int))
    for sentence in sentences:
        # Pad beginning of sentence
        context = [''] * (n - 1)
        for token in sentence:
            for i in range(len(context) + 1):
                frequencies[tuple(context[i:])][token] += 1
            if context:
                context.pop(0)
                context.append(token)
    return frequencies
