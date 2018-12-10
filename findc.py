import nltk
import numpy as np
import sklearn.decomposition
from nltk.corpus import brown

import summarizer_modules

"""
Computes c_0 as part of Arora's embeddings
From a subsection of the brown corpus
"""
# Get corpus
nltk.download('comtrans')
sents = brown.sents(categories=['news', 'editorial', 'reviews'])
print(sents)

# word2vec

#Form vector matrix
matrix = np.stack([summarizer_modules.str2vec_avg(' '.join(sent), args={'vectors': 'word2vec'}) for sent in sents],
                  axis=1)
#Perform pca
pca = sklearn.decomposition.PCA(n_components=5)
c0 = pca.fit_transform(matrix)
np.save('data/c0_word2vec', c0)

# Glove
matrix = np.stack([summarizer_modules.str2vec_avg(' '.join(sent), args={'vectors': 'glove'}) for sent in sents], axis=1)
# Perform pca
pca = sklearn.decomposition.PCA(n_components=5)
c0 = pca.fit_transform(matrix)
np.save('data/c0_glove', c0)
