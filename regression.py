import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

import summarizer_modules

str2vecs = [summarizer_modules.str2vec_avg, summarizer_modules.str2vec_arora, summarizer_modules.str2vec_doc2vec,
            summarizer_modules.str2vec_skipthought]

sentences = pickle.load(open('data/sent.pickle', 'rb'))
for i, f in enumerate(str2vecs):
    vecs = []
    y = []
    for sent, corr, words, emb, doc_emb, rouge in sentences:
        docvec = doc_emb[i]
        vec = emb[i] - docvec
        vecs.append(vec.T)
        y.append(rouge)

    x = np.stack(vecs, axis=0)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    lm = sm.OLS(y, sm.add_constant(x)).fit_regularized(alpha=.5)
    lm.save('data/regression_' + f.__name__ + ".pickle")
