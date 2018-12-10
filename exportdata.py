import ducpreprocess
import rouge as myrouge
import summarizer_modules
from ducpreprocess import *

"""Saves a variety of data to files which are loaded and analyzed by the jupyter notebook"""

corpus = ducpreprocess.load2001()

str2vecs = [summarizer_modules.str2vec_avg, summarizer_modules.str2vec_arora, summarizer_modules.str2vec_doc2vec,
            summarizer_modules.str2vec_skipthought]
args = {'vectors': 'word2vec'}

if not os.path.isfile('data/sent.pickle'):
    # Compile sentences
    sentences = []
    for v in corpus.values():
        if v.type == 1:
            doc = ' '.join(v.text)
            # doc_emb = [v.emb[f.__name__][-1] for f in str2vecs]
            doc_emb = [
                summarizer_modules.normalizeVec(np.mean(np.stack([v.emb[str2vec.__name__][:-1]], axis=1), axis=0)) for
                str2vec in str2vecs]
            for i, sent in enumerate(v.text):
                if not v.optimal_labels[i]:
                    emb = [v.emb[str2vec.__name__][i] for str2vec in str2vecs]
                    corr = [np.inner(doc_emb[i], emb[i]) for i in range(len(str2vecs))]
                    words = len(sent.split(' '))
                    # rouge = main.pyrouge.score_summary(main.getPyrougeSum([[sent]]), main.getPyrougeRef([v.summaries]), force_hong=True)
                    rouge = myrouge.compute_rouge(sent, [' '.join(x) for x in v.summaries[0]])[0]

                    sentences.append((sent, corr, words, emb, doc_emb, rouge))

    pickle.dump(sentences, open('data/sent.pickle', "wb"))
else:
    sentences = pickle.load(open('data/sent.pickle', 'rb'))
print("Completed sent embeddings")

if not os.path.isfile('data/docs.pickle'):
    # Compile sentences
    docs = []
    for k, v in corpus.items():
        if v.type == 1:
            doc = ' '.join(v.text)
            doc_emb = [v.emb[f.__name__][-1] for f in str2vecs]
            sentences = []
            for i, sent in enumerate(v.text):
                emb = [v.emb[f.__name__][i] for f in str2vecs]
                corr = [np.inner(doc_emb[i], emb[i]) for i in range(len(str2vecs))]
                print(corr)
                words = len(sent.split(' '))
                ref = v.optimal_labels[i]
                sentences.append((sent, corr, words, emb, doc_emb))
            sent_embs = [x[3] for x in sentences]
            docs.append((k, doc, doc_emb, sentences, sent_embs))
    pickle.dump(docs, open('data/docs.pickle', "wb"))
else:
    docs = pickle.load(open('data/docs.pickle', 'rb'))
print("Completed sent embeddings")

# Compile ref

ref = []
if not os.path.isfile('data/ref.pickle'):
    for v in corpus.values():
        if v.type == 1:
            doc = ' '.join(v.text)
            doc_emb = [v.emb[f.__name__][-1] for f in str2vecs]
            for i, sent in enumerate(v.text):
                if v.optimal_labels[i]:
                    emb = [x[i] for x in v.emb.values()]
                    corr = [np.inner(doc_emb[i], emb[i]) for i in range(len(str2vecs))]
                    words = len(sent.split(' '))
                    ref.append((sent, corr, words, emb, doc_emb))
    pickle.dump(ref, open('data/ref.pickle', "wb"))
else:
    ref = pickle.load(open('data/ref.pickle', 'rb'))
print("Completed ref embeddings")

if not os.path.isfile('data/matrix_str2vec_avg.txt'):
    for i, f in enumerate(str2vecs):
        vecs = []
        rouges = []
        for sent, corr, words, emb, doc_emb, rouge in sentences:
            docvec = doc_emb[i]
            vec = emb[i] - docvec
            vecs.append(vec.T)
            rouges.append(rouge)

        data = np.stack(vecs, axis=0)
        data = np.concatenate([np.asarray(rouges).reshape((9806, 1, 1)), data], axis=1)
        np.savetxt('matrix_' + f.__name__, data)

"""
if not os.path.isfile('data/sne.pickle'):
    # Compile sne
    sent_emb = [x[3][1] - x[4][1] for x in sentences]
    ref_emb = [x[3][1] - x[4][1] for x in ref]
    
    total_emb = sent_emb + ref_emb
    
    tsne = TSNE(n_jobs=4)
    total_sne = tsne.fit_transform(np.asarray(total_emb))
    
    sent_sne = total_emb[:len(sent_emb)]
    ref_sne = total_emb[len(sent_emb):]
    
    # Remove duplicated from doc_sne
    #As numpy arrays are hashable
    
    
    sne = (total_emb, total_sne)
    pickle.dump(sne, open('data/sne.pickle', "wb"))
"""
