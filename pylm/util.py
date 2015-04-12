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

def mle_pdist(fdist):
    """
    Create maximum likelihood probability distribution out of frequency
    distribution.

    Params:
        fdist: [dict] A dictionary mapping from keys to the frequencies of the
            keys.

    Returns:
        pdist: [dict] A dictionary mapping from keys to probabilities of the
            keys.
    """
    pdist = defaultdict(int)
    total_freqs = sum(fdist.values())
    for key, freq in fdist.items():
        pdist[key] = freq/total_freqs
    return pdist

def mle_cpd(cfd):
    """
    Create maximum likelihood probability distribution out of conditional
        frequency distribution.

    Params:
        cfd: [dict] Maps conditions to fdists.

    Returns:
        pdist: [dict] A dictionary mapping from conditions to pdists.
    """
    cpd = defaultdict(lambda: defaultdict(int))
    for condition, fdist in cfd.items():
        cpd[condition] = mle_pdist(fdist)
    return cpd
