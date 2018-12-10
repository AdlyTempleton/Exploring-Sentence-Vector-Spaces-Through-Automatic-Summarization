import functools
import itertools
import pickle
import random
import sys
from copy import copy

import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import brown
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

if not ('-nomodels' in sys.argv):
    # nltk.download("brown")
    # nltk.download("LancasterStemmer")
    doc2vec = Doc2Vec.load("data/doc2vec_conv.bin")
    word_vectors_word2vec = KeyedVectors.load_word2vec_format("data/vectors.bin", binary=True, unicode_errors="ignore",
                                                              limit=200000)
    word_vectors_glove = KeyedVectors.load_word2vec_format('data/vectors_glove.bin', binary=True,
                                                           unicode_errors="ignore", limit=200000)

    # Load fdist from file
    # fdist = nltk.FreqDist()
    # with open('data/ANC-written-count.txt') as freq_file:
    #    for line in csv.reader(freq_file, delimiter="\t"):
    #        fdist[line[0]] = int(line[3])
    #print(fdist.most_common(10))


    fdist = nltk.FreqDist(brown.words())

labels = []


def selector_max_sim(matched_sents, document, str2vec, wc, args={}):
    """Computes potential summaries from selector_near_then_bf and selector_greedy, and chooses the one with the highest cosine similarity with the document
    """
    candidates = [selector_near_then_bf(matched_sents, document, str2vec, wc, args),
                  selector_greedy(matched_sents, document, str2vec, wc, args)]
    vecs = [[[vec for s, vec in matched_sents if s == sent][0] for sent in candidate] for candidate in candidates]

    avg_vecs = [np.mean(np.stack(c), axis=0) for c in vecs]
    avg_vecs_scores = [np.inner(args['docvec'], vec) for vec in avg_vecs]
    print(avg_vecs_scores)
    labels.append(avg_vecs_scores[1] - avg_vecs_scores[0])
    return candidates[np.argmax(avg_vecs_scores)]

def selector_near_then_bf(matched_sents, document, str2vec, wc, args={}):
    """Takes the top ~15 sentences by cosine similarity, and finds the appropriate-sized group which maximizes cosine similarity"""
    docvec = args['docvec']
    scores = [np.inner(x[1], docvec) for x in matched_sents]
    threshold = sorted(scores)[-35]
    sent_pool = [sent for sent, score in zip(matched_sents, scores) if score >= threshold]
    combinations = itertools.chain.from_iterable((list(itertools.combinations(sent_pool, i)) for i in range(1, 7)))

    def meets_word_requirements(sents):
        sent_lens = [len(s[0].split()) for s in sents]
        max_sent = max(sent_lens)
        return sum(sent_lens) > 100 and sum(sent_lens) - max_sent < 100

    combinations = [x for x in combinations if meets_word_requirements(x)]
    avg_vecs = [np.mean(np.stack([vec for _, vec in c]), axis=0) for c in combinations]
    avg_vecs_scores = [np.inner(docvec, vec) for vec in avg_vecs]
    redundancies = [np.mean([np.inner(vec1, vec2) for (sent1, vec1), (sent2, vec2) in itertools.combinations(sents, 2)])
                    for sents in combinations]
    sents = combinations[np.argmax(avg_vecs_scores)]
    print(np.std(avg_vecs_scores))
    return sorted([sent for sent, vec in sents], key=len)


def selector_optimize_against_redundancy(matched_sents, document, str2vec, wc, args={}):
    """Takes the top ~15 sentences by cosine similarity, and finds the appropriate-sized group to minimize redundancy"""
    docvec = args['docvec']
    scores = [np.inner(x[1], docvec) for x in matched_sents]
    threshold = sorted(scores)[-15]
    sent_pool = [sent for sent, score in zip(matched_sents, scores) if score >= threshold]
    combinations = itertools.chain.from_iterable((list(itertools.combinations(sent_pool, i)) for i in range(1, 7)))

    def meets_word_requirements(sents):
        sent_lens = [len(s[0].split()) for s in sents]
        max_sent = max(sent_lens)
        return sum(sent_lens) > 100 and sum(sent_lens) - max_sent < 100

    combinations = [x for x in combinations if meets_word_requirements(x)]
    avg_vecs = [np.mean(np.stack([vec for _, vec in c]), axis=0) for c in combinations]
    avg_vecs_scores = [np.inner(docvec, vec) for vec in avg_vecs]
    redundancies = [np.mean([np.inner(vec1, vec2) for (sent1, vec1), (sent2, vec2) in itertools.combinations(sents, 2)])
                    for sents in combinations]
    sents = combinations[np.argmin(redundancies)]
    # sents = combinations[np.argmax(avg_vecs_scores)]
    print(np.std(avg_vecs_scores))
    return sorted([sent for sent, vec in sents], key=len)

def selector_pca(matched_sents, document, str2vec, wc, args={}):
    """Computes the principal components of the embeddings of the documents, then selects the sentence closest to each one"""
    vecs = [x[1] for x in matched_sents]
    pca = PCA()
    pca.fit(vecs)
    transformed = pca.transform(vecs)
    components = pca.components_
    r = []
    total_words = 0
    for component in components:
        max_sent, max_vec = max(matched_sents, key=lambda v: abs(np.inner(v[1], component)))
        new_words = len(max_sent.split())
        r.append(max_sent)
        total_words += new_words
        if total_words > wc:
            break
    return reorder_words(matched_sents, r)


def selector_near_then_random(matched_sents, document, str2vec, wc, args={}):
    """Baseline selector which selects the first sentence as near, then selects the rest randomly"""
    docvec = args['docvec']
    max_sent, max_vec = max(matched_sents, key=lambda sent: np.inner(sent[1], docvec))
    index = [i for i, e in enumerate(matched_sents) if e[0] == max_sent][0]

    scores = [random.random() for _ in range(len(matched_sents))]
    scores[index] = 2
    return select_from_scores(matched_sents, scores, wc, args)

def selector_pca_2(matched_sents, document, str2vec, wc, args={}):
    """Transforms all sentence vectors via PCA, and then selects the ones with the highest 1st dimension, then 2nd dimension, etc"""
    vecs = [x[1] for x in matched_sents]
    pca = PCA()
    pca.fit(vecs)
    transformed = pca.transform(vecs)
    matched_sents_transformed = [(sent, trans) for ((sent, vec), trans) in zip(matched_sents, transformed)]

    r = []
    total_words = 0
    for i in range(len(pca.components_)):
        max_sent, max_vec = max(matched_sents_transformed, key=lambda v: abs(v[1][i]))
        new_words = len(max_sent.split())
        r.append(max_sent)
        total_words += new_words
        if total_words > wc:
            break
    return reorder_words(matched_sents, r)


def selector_random(matched_sents, document, str2vec, wc, args={}):
    """Baseline selector to select sentences at random"""
    r = []
    total_words = 0
    while True:
        selected = random.choice(matched_sents)
        if not selected[0] in r:
            r.append(selected[0])
            total_words += len(selected[0].split())

        if total_words > wc:
            break
    return reorder_words(matched_sents, r)


# Code adapted from Sumy
# https://github.com/miso-belica/sumy/
def selector_lexrank_continuous(matched_sents, document, str2vec, wc, args={}):
    threshold = 0.1
    epsilon = 0.1
    shift = 0.2

    stop = set(stopwords.words('english'))
    n = len(matched_sents)
    matrix = np.zeros((n, n))
    degrees = np.zeros((n,))

    for row, (sent1, vec1) in enumerate(matched_sents):
        for col, (sent2, vec2) in enumerate(matched_sents):
            matrix[row, col] = max(np.inner(vec1, vec2) - shift, 0)

    # Normalize
    for row in range(n):
        matrix[row] = matrix[row] / sum(matrix[row])

    # Power iteration
    transposed_matrix = matrix.T
    p_vector = np.array([1.0 / n] * n)
    lambda_val = 1.0

    while lambda_val > epsilon:
        next_p = np.dot(transposed_matrix, p_vector)
        lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
        p_vector = next_p
    scores = p_vector
    return select_from_scores(matched_sents, scores, wc, args)


def selector_lexrank_discrete(matched_sents, document, str2vec, wc, args={}):
    threshold = 0.25
    epsilon = 0.1

    stop = set(stopwords.words('english'))
    n = len(matched_sents)
    matrix = np.zeros((n, n))
    degrees = np.zeros((n,))

    for row, (sent1, vec1) in enumerate(matched_sents):
        for col, (sent2, vec2) in enumerate(matched_sents):
            matrix[row, col] = 1 if np.inner(vec1, vec2) > threshold else 0

    # Normalize
    for row in range(n):
        matrix[row] = matrix[row] / sum(matrix[row])

    # Power iteration
    transposed_matrix = matrix.T
    p_vector = np.array([1.0 / n] * n)
    lambda_val = 1.0

    while lambda_val > epsilon:
        next_p = np.dot(transposed_matrix, p_vector)
        lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
        p_vector = next_p
    scores = p_vector
    return select_from_scores(matched_sents, scores, wc, args)

def selector_complete(matched_sents, document, str2vec, wc, args={}):
    """Baseline selector to select entire document"""
    return [x[0] for x in matched_sents]


diffs = []


#import main
def selector_greedy(matched_sents, document, str2vec, wc, args={}):
    """At every iteration, selects the sentence vector such that the new average summary is more similar to the document"""
    # Copy of matched_sents, as we will be removing vectors from it
    matched_sents_copy = copy(matched_sents)
    matched_sents = copy(matched_sents)
    docvec = args['docvec']
    total_words = 0

    r=[]
    r_text=[]

    # Use max as argmax
    max_sent, max_vec = sorted(matched_sents, key=lambda sent: np.inner(sent[1], docvec))[-1]
    max_sent, max_vec = sorted(matched_sents, key=lambda sent: np.inner(sent[1], docvec))[-1]

    r.append(max_vec)
    r_text.append(max_sent)
    total_words += len(max_sent.split())
    avg_vec = max_vec
    index = [i for i, e in enumerate(matched_sents) if e[0] == max_sent][0]

    matched_sents.pop(index)
    while len(matched_sents) > 0:
        n = float(len(r))
        # argmax, where the second argument of np.inner is an expression for the new average vector
        r_matrix = np.stack(r, axis=0).T
        max_sent, max_vec = None, None
        if 'greedy_avg' in args and args['greedy_avg']:
            max_sent, max_vec = max(matched_sents, key=lambda sent: np.inner(docvec, normalizeVec(
                np.mean(np.append(r_matrix, (sent[1].reshape(r_matrix.shape[0], 1)), axis=1), axis=1))))
        else:
            max_sent, max_vec = max(matched_sents,
                                    key=lambda sent: np.inner(docvec, str2vec(' '.join(r_text + [sent[0]]), args=args)))

        #

        #Check that we are under wc

        total_words += len(max_sent.split())
        avg_vec = avg_vec + avg_vec * (n / (n + 1.0)) + max_vec[1] * (1.0 / (n + 1))
        index = [i for i, e in enumerate(matched_sents) if e[0] == max_sent][0]
        matched_sents.pop(index)
        r_text.append(max_sent)
        r.append(max_vec)

        if (total_words > wc):
            break
    #Reorder r_text to match original document order
    return reorder_words(matched_sents_copy, r_text)


def reorder_words(matched_sents, r):
    # NO-OP due to issues w summary length
    # Simple implementation is bad for multi-document summarization in general
    # And reordering does not increase ROUGE scores significantly
    # TODO-advanced reorder algorithm
    # r_ordered = []
    # for sent in matched_sents:
    #    if sent[0] in r:
    #
    return r


def selector_lead_baseline(matched_sents, document, str2vec, wc, args={}):
    """Baseline which selects the first sentences"""
    scores = list(range(0, -1 * len(matched_sents), -1))
    return select_from_scores(matched_sents, scores, wc, args)


def selector_near_redundancy(matched_sents, document, str2vec, wc, args={}):
    """A version of selector_near which also takes into account the redundancy between selector sentences
    Redundancy is correlated to cosine similarity, so we weight redundnacy based on the residuals of a regression (seen in jupyer notebook)
    """
    docvec = args['docvec']
    r = []
    r_vecs = []
    max_sent, max_vec = max(matched_sents, key=lambda sent: np.inner(sent[1], docvec))
    r.append(max_sent)
    r_vecs.append(max_vec)
    total_words = 0

    index = [i for i, e in enumerate(matched_sents) if e[0] == max_sent][0]
    matched_sents.pop(index)
    total_words += len(max_sent.split())
    while len(matched_sents) > 0:
        n = float(len(r))
        # argmax
        max_sent, max_vec = max(matched_sents,
                                key=functools.partial(selector_near_redundancy_score, r_vecs, docvec, str2vec))

        r_vecs.append(max_vec)
        r.append(max_sent)
        total_words += len(max_sent.split())
        index = [i for i, e in enumerate(matched_sents) if e[0] == max_sent][0]
        matched_sents.pop(index)
        if total_words > wc:
            break
    return r


def selector_near_redundancy_score(prev_vecs, docvec, str2vec, matched_sent):
    """Score function for selector_near_redundancy
    Uses a regression fitted on cosine similarity, and weights the residuals from that regression
    """
    curves = {str2vec_avg: np.poly1d([0.7468 + 0.2372 + 0.04282]), str2vec_arora: np.poly1d([0.8026, 0.3422, 0.02032]),
              str2vec_doc2vec: np.poly1d([0.7023, 0.186]), str2vec_skipthought: np.poly1d([0.7023, 0.186])}
    factor = 1
    sent, vec = matched_sent
    near_score = np.inner(docvec, vec)
    redundancy = np.mean([np.inner(vec, prev_vec) for prev_vec in prev_vecs])
    residual = redundancy - curves[str2vec](near_score)
    return near_score - residual * factor


def selector_near(matched_sents, document, str2vec, wc, args={}):
    """
    Basic selector which selects sentences with highest cos sim to docvec
    """
    # vector for entire document
    docvec = args['docvec']

    r=[]
    scores = [np.inner(x[1], docvec) for x in matched_sents]
    return select_from_scores(matched_sents, scores, wc)


def selector_far(matched_sents, document, str2vec, wc, args={}):
    """
    Basic selector which selects sentences with lowest cos sim to docvec
    Designed for use with GLOVE only
    """
    # vector for entire document
    docvec = str2vec(' '.join(document), args)

    r = []
    scores = [-1 * np.inner(x[1], docvec) for x in matched_sents]
    return select_from_scores(matched_sents, scores, wc)


def make_capped(selector):
    return functools.partial(selector_capped, selector)



def selector_capped(selector, matched_sents, document, str2vec, wc, args={}):
    """Wrapper on other selectors that prevents choosing sentences over 20 words"""
    matched_sents = [x for x in matched_sents if 10 < len(x[0].split()) < 30]
    return selector(matched_sents, document, str2vec, wc, args)


def selector_near_adjusted(matched_sents, document, str2vec, wc, args={}):
    """
    Selector which adjusts corr scores by a linear regression to favor shorter sentences
    Also incorporates a per-dimension regression score
    """
    docvec = args['docvec']

    r = []

    # Apply results of regression model
    diff_vecs = [x[1] - docvec for x in matched_sents]

    def regression_scores(vec):
        return 2.15 * vec[18] - 3.23 * vec[36] - 3.08 * vec[90] - 3.58 * vec[159] + 2.54 * vec[195] - 2.01 * vec[249]

    # regg_scores = [regression_scores(vec) for vec in diff_vecs]
    # From regression: .008921
    scores = [(np.inner(x[1], docvec) + .4 * np.log(len(x[0].split()))) for x in matched_sents]
    # scores = [(score + .2 * regg_score) for score, regg_score in zip(scores, regg_scores)]

    return select_from_scores(matched_sents, scores, wc)


if not ('-nomodels' in sys.argv):
    svm_avg = pickle.load(open('data/svm.pickle', 'rb'), encoding='latin1')
    svm_doc2vec = pickle.load(open('data/svm_doc2vec.pickle', 'rb'), encoding='latin1')


def selector_long(matched_sents, document, str2vec, wc, args={}):
    scores = [len(sent[0].split()) for sent in matched_sents]
    scores = [s if s < 45 else 0 for s in scores]
    return select_from_scores(matched_sents, scores, wc, args)

def selector_svm(matched_sents, document, str2vec, wc, args={}):
    """Selects sentences using a SVM, treating the problem as traditional text classification"""
    svm = svm_avg
    if 'svm' in args:
        if args['svm'] == 'doc2vec':
            svm = svm_doc2vec
    docvec = args['docvec']
    vecs = [x[1] - docvec for x in matched_sents]
    data = np.stack(vecs, axis=0)
    scores = [x[1] for x in svm.predict_proba(data)]
    return select_from_scores(matched_sents, scores, wc)

def select_from_scores(matched_sents, scores, wc, args={}):
    matched_sents_copy = list(matched_sents)
    r = []
    scores = [(score, len(sent[0].split())) for score, sent in zip(scores, matched_sents)]

    total_words = 0
    while True:
        sent, _ = matched_sents[max(range(len(matched_sents)), key=lambda i: scores[i][0])]
        r.append(sent)
        total_words += len(sent.split())
        index = [i for i, e in enumerate(matched_sents) if e[0] == sent][0]
        matched_sents.pop(index)
        if total_words > wc:
            break
    return reorder_words(matched_sents_copy, r)


stmodel = None;
stencoder = None;
if not ('-nomodels' in sys.argv or '-nost' in sys.argv):
    from lib.skipthoughts import skipthoughts
    stmodel = skipthoughts.load_model()
    stencoder = skipthoughts.Encoder(stmodel)


def str2vec_skipthought(s, args={}):
    return normalizeVec(stencoder.encode([s], verbose=False)[0])


def str2vec_doc2vec(s, args={}):
    words = s.split()
    return normalizeVec(doc2vec.infer_vector(words))


missing_words = 0

#Average vector for sentence using arora models
def str2vec_avg(s, args={}):
    #TODO: Align tokens with word2vec's tokens
    #Tokenize words
    words = s.split()

    word_vectors = word_vectors_word2vec
    if 'vectors' in args and args['vectors'] == 'glove':
        word_vectors = word_vectors_glove

    sum = np.zeros(300)
    for word in words:
        if word in word_vectors:
            #Compute weighting according to smooth inverse frequencies, from arora et al
            a = .00001
            freq = fdist.freq(word)
            sif = a / (a + freq)
            normvec = normalizeVec(word_vectors[word])
            sum = sum + (sif * normvec)
            # if freq == 0:
            #    print("Missing {}".format(word))
    #Subtract projection along c0
    # normalize result vector
    return normalizeVec(sum)


def str2vec_inv(s, args={}):
    # TODO: Align tokens with word2vec's tokens
    # Tokenize words
    words = s.split()

    word_vectors = word_vectors_word2vec
    if 'vectors' in args and args['vectors'] == 'glove':
        word_vectors = word_vectors_glove

    sum = np.zeros(300)
    for word in words:
        if word in word_vectors:
            vec = word_vectors[word]
            len = np.linalg.norm(vec)
            adj_vec = vec / len ** 2 if len != 0 else vec
            sum = sum + adj_vec
            # if freq == 0:
            #    print("Missing {}".format(word))
    # Subtract projection along c0
    # normalize result vector
    #return normalizeVec(sum)
    return normalizeVec(sum)




if not ('-nomodels' in sys.argv):
    c0_word2vec = np.load('data/c0_word2vec.npy')
    c0_glove = np.load('data/c0_glove.npy')


def str2vec_arora(string, args={}):
    c0 = c0_word2vec
    if 'vectors' in args and args['vectors'] == 'glove':
        c0 = c0_glove

    r = str2vec_avg(string, args)
    for i in range(1):  # range(c0.shape[1]):
        vec = c0[:, i]
        vec = np.reshape(vec, np.shape(r))
        r = r - (np.inner(r, vec) / np.inner(vec, vec)) * vec
    return normalizeVec(r)


import sklearn


def str2vec_arora_true(string, args={}):
    text = args['text']

    c0 = None
    if not 'c0' in args:
        vec_matrix = np.stack([str2vec_avg(' '.join(sent), args={'vectors': 'word2vec'}) for sent in text], axis=1)

        pca = sklearn.decomposition.PCA(n_components=1)
        c0 = pca.fit_transform(vec_matrix)

        args['c0'] = c0
    else:
        c0 = args['c0']


    r = str2vec_avg(string, args)
    for i in range(1):  # range(c0.shape[1]):
        vec = c0[:, i]
        vec = np.reshape(vec, np.shape(r))
        r = r - (np.inner(r, vec)/np.inner(vec, vec))*vec
    return normalizeVec(r)


def normalizeVec(vec):
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec

def selector_combined(matched_sents, document, str2vec, wc, args={}):
    results = [selector_cluster(matched_sents, document, str2vec, wc, args),
               selector_svm(matched_sents, document, str2vec, wc, args),
               selector_near(matched_sents, document, str2vec, wc, args),
               selector_greedy(matched_sents, document, str2vec, wc, args)]
    results = sum(results, [])
    scores = list(map(results.count, results))
    # Dummy
    results_matched_sents = [(x, 0) for x in results]
    return select_from_scores(results_matched_sents, scores, wc)

def selector_cluster(matched_sents, document, str2vec, wc, args={}):
    vecs = np.asarray([x[1] for x in matched_sents])
    # Start clustering algorithm

    n = 3

    # Increase clusters until we get enough words
    while True:
        labels = AgglomerativeClustering(n_clusters=min(n, len(matched_sents)), affinity='cosine',
                                         linkage='complete').fit_predict(vecs)
        # Actually, one less that the number of clusters, by zero-indexing
        num_clusters = max(labels)
        clusters = [[vecs[i] for (i, label) in enumerate(labels) if label == c] for c in range(num_clusters)]
        centers = [sum(c) / len(c) for c in clusters]
        r = []
        wl = 0
        for center in centers:
            max_sim = (-1, (None, None))
            for sent in matched_sents:
                if np.inner(sent[1], center) > max_sim[0]:
                    max_sim = (np.inner(sent[1], center), sent)
            r.append(max_sim[1][0])
            wl += len(max_sim[1][0].split())
            if wl > wc:
                return r
        n += 1


import statsmodels.api as sm
import pandas as pd

str2vecs = [str2vec_avg, str2vec_arora, str2vec_doc2vec, str2vec_skipthought]
regression_models = dict()


def selector_regression(matched_sents, document, str2vec, wc, args={}):
    if len(regression_models) == 0:
        for f in str2vecs:
            regression_models[f.__name__] = sm.load('data/regression_' + f.__name__ + ".pickle")

    model = regression_models[str2vec.__name__]

    x = np.stack([vec for _, vec in matched_sents], axis=0)
    x = pd.DataFrame(x)
    x = sm.add_constant(x)
    scores = model.predict(exog=x).values.tolist()
    return select_from_scores(matched_sents, scores, wc, args)
if __name__ == "__main__":
    print(str2vec_arora("How much work we have to do to talk to each other"))
