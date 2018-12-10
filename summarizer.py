import functools

import nltk
import numpy as np
from nltk.stem.porter import *
from sklearn.decomposition import PCA

import summarizer_modules


# nltk.download('punkt')
# nltk.download('brown')

def keywords(v, str2vec, selector, wc, args):
    stemmer = PorterStemmer()
    args['semeval'] = True
    matched_phrases = [(phrase, str2vec(phrase)) for phrase in v.candidates]
    # matched_phrases = [(phrase, emb) for phrase, emb in zip(v.candidates, v.emb[str2vec.__name__])]
    # Check for zero-vectors
    matched_phrases = [x for x in matched_phrases if not np.linalg.norm(x[1]) == 0.0]
    vecs = [x[1] for x in matched_phrases]
    if 'docvec_avg' in args and args['docvec_avg']:
        args['docvec'] = np.mean(np.stack(vecs, axis=1), axis=0)
    else:
        args['docvec'] = str2vec(' '.join(v.text))

    summary = selector(matched_phrases, v.text, str2vec, wc, args)
    excess_words = len(' '.join(summary).split()) - 100
    summary[-1] = ' '.join(summary[-1].split()[:-excess_words])
    print(len(' '.join(summary).split()))
    # assert len(summary) == len(set(summary))
    return [stemmer.stem(phrase) for phrase in summary]


def summarize(v, str2vec, selector, wc, args):
    """
    Creates a summary of a document

    :param v: A Corpus object
    :param str2vec: A function which turns documents into vectors
    With the following ordered parameters: string, args
    :param selector:
    A function which, given sentence embeddings, selects a summary
    With the following ordered parameters: matched_sents, document, str2vec, wc, args
        matched_sents: A list of tuples, (sentence, embedding)
        document: A document, as a list of sentences
        str2vec: A str2vec function, as above
        wc: The maximum word count of the summary
        args: Additional model-specific parameters
    :param wc: The maximum word count of the summary
    :return: A summary, as a list of sentences
    """
    matched_sents = []

    # Clone args, allow modification (ie str2vec_arora_true) to args from functions
    args = dict(args)

    # Add additional data
    args['text'] = v.text
    args['v'] = v

    # Vector for sentences
    if str2vec.__name__ in v.emb.keys() and not args.get('skip_cache', False):
        matched_sents = [(sent, emb) for sent, emb in zip(v.text, v.emb[str2vec.__name__])]
    else:
        matched_sents = [(sent, str2vec(sent, args)) for sent in v.text]
    # Check for zero-vectors
    matched_sents = [x for x in matched_sents if not np.linalg.norm(x[1]) == 0.0]
    vecs = [x[1] for x in matched_sents]




    # PCA transform
    if 'pca' in args and args['pca']:
        pca = PCA(n_components=.5, whiten=True)
        vecs = [x[1] for x in matched_sents]
        pca.fit(vecs)
        vecs = pca.transform(vecs)
        matched_sents = [(sent, trans) for ((sent, vec), trans) in zip(matched_sents, vecs)]
        str2vec = wrap_for_pca(str2vec, pca)

    if 'docvec_avg' in args and args['docvec_avg']:
        args['docvec'] = summarizer_modules.normalizeVec(np.mean(np.stack(vecs, axis=1), axis=1))
    else:

        if str2vec.__name__ in v.emb.keys() and not args.get('skip_cache', False):
            args['docvec'] = v.emb[str2vec.__name__][-1]
        else:
            args['docvec'] = str2vec(' '.join(v.text), args)
    # Check for duplicates


    summary = selector(matched_sents, v.text, str2vec, wc, args)
    #Cut to 100 words
    excess_words = len(' '.join(summary).split()) - 100
    summary[-1] = ' '.join(summary[-1].split()[:-excess_words])
    #assert len(summary) == len(set(summary))
    return summary


def wrap_for_pca(str2vec, pca):
    return functools.partial(str2vec_from_pca, str2vec, pca)


def str2vec_from_pca(str2vec, pca, s, args={}):
    vec = str2vec(s, args)
    vec = vec.reshape(1, -1)
    return pca.transform(vec)

if __name__ == "__main__":
    raw = open('input.txt').read()

    # Split sentences
    sents = nltk.sent_tokenize(raw)

    # Frequency distribution from brown corpus

    # Load google news word vectors
    print("Loaded data")
