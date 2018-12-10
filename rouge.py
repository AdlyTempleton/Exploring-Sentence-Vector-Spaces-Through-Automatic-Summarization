from collections import Counter

import numpy as np
from nltk.corpus import wordnet
from nltk.stem.porter import *

stemmer = PorterStemmer()


def get_exception(word):
    exceptions = wordnet._exception_map
    for d in exceptions.values():
        if word in d:
            return d[word][0]
    return False


def stem_with_exception(word):
    exception_lookup = get_exception(word)
    if not exception_lookup:
        return stemmer.stem(word)
    else:
        return exception_lookup


def morphy_with_exception(word):
    r = wordnet.morphy(word)
    if r == None:
        r = word
    return r


def compute_rouge(summ, refs):
    summ = preprocess(summ)
    refs = [preprocess(ref) for ref in refs]
    refs = [' '.join(x.split()[:min(100, len(x.split()))]) for x in refs]
    summ = ' '.join(summ.split()[:min(100, len(summ.split()))])
    return (compute_rouge_n(summ, refs, 1), compute_rouge_n(summ, refs, 2))


def preprocess(s):
    return ' '.join([stem_with_exception(w) for w in s.split()])


def compute_rouge_n(summ, refs, n):
    scores = []
    for ref in refs:
        ngram_summ = n_grams_counted(summ, n)
        ngram_ref = n_grams_counted(ref, n)
        ngram_match = ngram_summ & ngram_ref
        match_count = sum([count for item, count in ngram_match.items()])
        ref_count = sum([count for item, count in ngram_ref.items()])
        scores.append((match_count * 1.000) / ref_count)
    return np.mean(scores)


def n_grams_counted(s, n):
    return Counter(n_grams(s, n))

def n_grams(s, n):
    words = s.split()
    return [tuple(words[start:start + n]) for start in range(len(words) - n + 1)]


if __name__ == "__main__":
    text = "Hello world this is a very very good day"
    print(n_grams(text, 4))
