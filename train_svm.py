from sklearn.svm import SVC

import ducpreprocess
import summarizer_modules
from ducpreprocess import *

# Trains and saves a SVM model
corpus = ducpreprocess.load2001()

data, labels = ducpreprocess.get_sent_matrix(corpus, diff=True)
model1 = SVC(probability=True, C=.2).fit(data, labels)

data_dev, labels_dev = ducpreprocess.get_sent_matrix(corpus, type=2, diff=True)
print(model1.score(data_dev, labels_dev))
pickle.dump(model1, open('svm.pickle', "wb"))

data, labels = ducpreprocess.get_sent_matrix(corpus, str2vec=summarizer_modules.str2vec_doc2vec, diff=True)
model2 = SVC(probability=True, C=.2).fit(data, labels)

data_dev, labels_dev = ducpreprocess.get_sent_matrix(corpus, str2vec=summarizer_modules.str2vec_doc2vec, diff=True,
                                                     type=2)
print(model2.score(data_dev, labels_dev))

pickle.dump(model2, open('svm_doc2vec.pickle', "wb"))

data, labels = ducpreprocess.get_sent_matrix(corpus, str2vec=summarizer_modules.str2vec_skipthought, diff=True)
model2 = SVC(probability=True, C=.2).fit(data, labels)

data_dev, labels_dev = ducpreprocess.get_sent_matrix(corpus, str2vec=summarizer_modules.str2vec_doc2vec, type=2,
                                                     diff=True)
print(model2.score(data_dev, labels_dev))

pickle.dump(model2, open('svm_skipthought.pickle', "wb"))
