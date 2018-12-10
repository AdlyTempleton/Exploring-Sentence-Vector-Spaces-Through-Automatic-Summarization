import os
from collections import Counter

from bs4 import BeautifulSoup
from nltk.stem.porter import *

from summarizer_modules import str2vec_arora

"""
A collection of functions related to reading in the dataset and other similar misc tasks
"""

def fetch_filenames(base):
    """
    Recursively returns all files exactly two levels down from a given filepath

    :param base: Base filepath to start search
    :return: A list of filenames
    """
    docs = [[base + x + '/' + d for d in os.listdir(base + x)] for x in os.listdir(base) if os.path.isdir(base + x)]
    t = []
    for i in range(len(docs)):
        for j in range(len(docs[i])):
            t.append(docs[i][j])
    docs = t
    return docs

def add_summary_2004(corpus, filename):
    """
    Version of add_summary adapted for the 2004 dataset
    :param corpus: Common corpus object
    :param filename: Filename containing a summary
    """
    with open(filename, 'r') as file:
        # Get key from the filename
        key = cid(filename.split('/')[-1].split('.')[0])
        #2004 summaries are simple txt files
        corpus[key].summaries.append(file.read())

def add_summary(corpus, file, type):
    """
    Given a summary file, add it to the appropriate document in the corpus

    :param corpus: The standard composite corpus opposite
    :param file: The filename of the summary
    :param type: The document type of the summary (test, training, etc)
    :param keyfunction: A function which takes a beautifulsoup obkect and reutrns a key
    :return:
    """

    # Open XML parser
    soup = BeautifulSoup(open(file), 'html.parser')

    for sum in soup.find_all('sum'):
        # Find the document key
        # Replacement is a hacky solution to a common typo in the dataset metadata
        key = sum['docref'].replace("SMN", "SJMN")
        corpus_value = corpus[key]

        # Append summary text to appropriate document
        corpus_value.summaries.append(sum.string)

        # Check for mismatched types
        if not corpus_value.type == 0 and not corpus_value.type == type:
            print(
                "Fighting over document type: Original type: {}. New type: {}. Key: {}. Cluster: {} New file name: {}".format(
                    corpus_value.type, type, key, corpus[key][3], file))
        # (re)set document type
        corpus_value.type = type


def process_data2004():
    """
    :return: A clusters object
    A dictionary of cluster ids
    Each mapping to a CorpusEntry2004

    Note that this method returns 'lists' which are actually strings
    To be processed into lists of sentences by a later method
    """
    #Fetches filenames from a nested directory structure
    docs = fetch_filenames('datasets/duc2004/docs/')
    summaries_base_dir = 'datasets/duc2004/models/'
    summaries = [summaries_base_dir + d for d in os.listdir(summaries_base_dir)]
    corpus2004 = dict()
    for d in docs:

        soup = BeautifulSoup(open(d), 'html.parser')
        key = cid(d.split('/')[-2])

        if not key in corpus2004.keys():
            #create skeleton entry
            value = CorpusEntry2004(key)
            value.type = 3
            corpus2004[key] = value

        # Raw text from XML tree
        text = soup.doc.text

        #Append text
        entry = corpus2004[key]
        entry.text_by_doc.append(text)

    # Add summaries
    for f in summaries:
        add_summary_2004(corpus2004, f)
    return corpus2004


def process_data():
    """
    First step in data loading pipeline

    The corpus object itself is a dict mapping strings to 4-tuples (technically lists)
    Key: The id of the document
    [0]: The full text of the document
    [1]: A list of summaries, possibly empty
    [2]: Status of the document (0 = unassigned, 1 = training, 2 = dev, 3 = test, 4=DUC 2002
    [3]: Cluster of the document

    :return: The common corpus object, with document and summary info, minus some processing
    """

    #Find all document filenames
    docs = fetch_filenames('rdatasets/duc2001/docs/')

    #List comprehension to find name of perdocs
    docs_training_perdocs = [x + '/perdocs' for x in fetch_filenames('datasets/duc2001/data/training/') if
                             not '/docs' in x and os.path.isdir(x)]
    docs_testing_perdocs = [x + '/perdocs' for x in fetch_filenames('datasets/duc2001/data/test/') if
                            not '/docs' in x and os.path.isdir(x)]

    corpus = dict()
    for d in docs:
        # Prevent duplicates
        #Another quirk of dataset format
        if d[-2:] == ".S":
            continue
        # First three lines are non-XML
        soup = BeautifulSoup(open(d), 'html.parser')
        # Hacky solution to a frequent typo
        key = soup.docno.string.replace('\n', '').replace(' ', '').replace("SMN", "SJMN")
        #Find cluster from subdirectory in filepath
        cluster = d.split('/')[-2]
        #Raw text from XML tree
        text = soup.doc.text
        #Check for duplicate documents
        if key in corpus:
            print("Error duplicate document:\n{}".format(key))
        # Add entry to corpus
        #Note that it will clear duplicate documents
        v = CorpusEntry2001(key)
        v.text, v.cluster, v.type = text, cluster, 0
        corpus[key] = v

    #Add summaries
    for f in docs_testing_perdocs:
        add_summary(corpus, f, 3)

    for f in docs_training_perdocs:
        add_summary(corpus, f, 1)
    return corpus


def mark_dev(corpus):
    """
    Randomly marks half of the testing documents as validation
    Seeded on the cluster name, so consistent between runs

    :param corpus: Common corpus object
    """
    for k, v in corpus.items():
        if v.type == 3:
            # Same values for each cluster
            # Distinguish between 2004 and 2001 corpuses
            key = cid(v.cluster) if not isinstance(v, CorpusEntry2004) else v.key
            random.seed(123 * key + 1554)
            if random.random() > .5:
                v.type = 2


def data_statistics(corpus):
    """
    Prints a variety of 'sanity check' statistics after processing

    :param corpus: Common corpus object
    """
    #Store counts
    sumCount = 0
    blankCount = 0
    typeCount = [0, 0, 0, 0]


    for key in sorted(list(corpus.keys())):
        sumCount += len(corpus[key].summaries)
        if len(corpus[key].summaries) == 0:
            blankCount += 1
        typeCount[corpus[key].type] += 1

    print("{} documents, {} summaries".format(len(corpus), sumCount))
    print("{} documents without a summary".format(blankCount))
    print(
        "{} test documents, {} dev documents, {} training documents ({} unassigned)".format(typeCount[3], typeCount[2],
                                                                                            typeCount[1], typeCount[0]))


def preprocess_text(text, junk_filter=True):
    """
    Processes a single 'line' of text into a computer readable format

    :param text: A string of unprocessed text
    :return: A list of sentences, with the text inside transformed
    """

    # Based special char replacement
    text = text.replace("\n", " ")
    text = text.replace("\\", "")

    #Break text into sentences(abbreviation-aware)
    text = nltk.sent_tokenize(text)

    #Heuristics to strip out titles
    # All sentences with one of these strings is removed
    title_strings = ['WALL STREET JOURNAL (J', '(AP)', 'PHOTO', 'AP-NR', 'Home Edition', 'TIMES STAFF WRITER', 'WSJ',
                     'San Jose Mercury News', 'Morning Final     SJ', '         ']

    if junk_filter:
        for title_string in title_strings:
            removed_sentences = []
            for sent in text:
                if title_string in sent:
                    removed_sentences.append(sent)
                # Short sentences are almost always junk
                if len(sent) < 10:
                    removed_sentences.append(sent)
            for sent in removed_sentences:
                text.remove(sent)

    #Similarily, very long sentences are almost always junk (ie. tables)
    text = [s for s in text if not len(s.split(' ')) > 100]

    # Lowercase, for more reliable word embeddings
    text = [t.lower() for t in text]
    return text


def preprocess_texts2004(corpus):
    """
    Preprocessing on the entire common corpus object
    Is based upon preprocess_text, which it calls repeatedly

    :param corpus: Common corpus object
    """
    for v in corpus.values():
        # Preprocess document text
        v.text_by_doc = [preprocess_text(t, junk_filter=False) for t in v.text_by_doc]
        # Preprocess summaries (Note that summary raw text is cleaner, without metadata
        v.summaries = [preprocess_text(t, junk_filter=False) for t in v.summaries]
        v.text = sum(v.text_by_doc, [])

def preprocess_texts(corpus):
    """
    Preprocessing on the entire common corpus object
    Is based upon preprocess_text, which it calls repeatedly

    :param corpus: Common corpus object
    """
    for v in corpus.values():
        #Preprocess document text
        v.text = preprocess_text(v.text)
        # Chop off title/metadata
        v.text = v.text[1:]
        #Preprocess summaries (Note that summary raw text is cleaner, without metadata
        v.summaries = [preprocess_text(t) for t in v.summaries]


def check_extract(corpus):
    """
    Usused code to detect that all sentences in extract are from document

    :param corpus: Common corpus object
    """
    c = 0
    t = 0
    for v in corpus.values():
        for sum in v.summaries:
            for sum_sent in sum:
                if not sum_sent in v.text:
                    c += 1
                    print(v.text)
                    print(sum_sent)
                t += 1
    print("{} missing sentences out of {}".format(c, t))


def n_grams(s, n):
    """Finds all n-grams in a string
    :param s: String in which to search
    :param n: Length of n-grams to find
    :return: A list of n-grams, each of which is a list of words"""
    words = s.split()
    return [words[start:start + n] for start in range(len(words) - n + 1)]


def load_data_semeval(directory, type):
    """Obsolete method for loading semeval dataset"""
    stemmer = PorterStemmer()
    corpus_semeval = dict()

    docs = os.listdir(directory)
    docs = [d for d in docs if '.txt.final' in d]
    for doc in docs:
        key = doc.split('.')[0]
        text = open(directory + '/' + doc, encoding='utf-8').read()
        text = text.replace("\n", " ")
        # Cut off references and header
        text = text[text.find('1.'):]
        text = text[:text.rfind('[1]')]
        sents = nltk.sent_tokenize(text)

        entry = SemEvalEntry(key)
        entry.text = sents
        candidates = [[n_grams(sent, i) for i in range(1, 6)] for sent in sents]
        # flatten
        candidates = sum([sum(x, []) for x in candidates], [])
        stopwords = nltk.corpus.stopwords.words('english')
        # Remove candidates starting or ending with stopwords
        candidates = [candidate for candidate in candidates if
                      candidate[0] not in stopwords and candidate[-1] not in stopwords]
        # Join to strings
        candidates = [' '.join(candidate) for candidate in candidates]
        # Remove symbols
        candidates = [candidate for candidate in candidates if all([x.isalnum() or x.isspace() for x in candidate])]

        count = Counter(candidates)
        entry.candidates = [x[0] for x in count.most_common(100)]
        entry.type = type
        corpus_semeval[key] = entry
    keywords_lines = open(directory + '/trial.combined.stem.final').readlines()
    for keyword_line in keywords_lines:
        key, value = (x.strip() for x in keyword_line.split(':'))
        entry = corpus_semeval[key]
        entry.keywords = value.split(',')
    return corpus_semeval


# def complete_data_semeval():
#    return load_data_semeval('semeval/trial', 3)

def complete_data_2004():
    """Complete pipeline for loading 2004 data"""

    corpus2004 = process_data2004()
    mark_dev(corpus2004)

    preprocess_texts2004(corpus2004)
    return corpus2004


def complete_data_2001():
    """
    Runs processing pipeline for 2001 corpus

    :return: Common corpus object
    """
    nltk.download('punkt')
    corpus = process_data()
    mark_dev(corpus)
    data_statistics(corpus)
    preprocess_texts(corpus)
    return corpus

def cid(str):
    """
    Helper method to get numeric id from cluster filepath string

    ex. "d03a" -> 3
    :param str: Document id as a string
    :return: Document id as an integer
    """

    return int(''.join([c for c in str if c.isdigit()]))


def cluster_data(corpus):
    """Prepares data for multi-document summarization and groups documents by cluster

    :param corpus: Common corpus object
     Simply the same object from single-document-summarization corpus
    :return: A clusters object, mapping to CorpusEntry2001Cluster
    """
    clusters = dict()
    # Make basic framework for all cluster ids
    #Note the built-in overwriting
    for v in corpus.values():
        clusters[cid(v.cluster)] = CorpusEntry2001Cluster(cid(v.cluster))


    for v in corpus.values():
        cluster = clusters[cid(v.cluster)]

        #Add text to cluster text
        cluster.text += v.text

        #Check for type fighting
        if cluster.type != -1 and cluster.type != v.type:
            print("Fight over type of cluster: {} from {} to {}".format(v.summaries, cluster.type, v.type))
            if v.type == 0:
                continue
        #Update type
        cluster.type = v.type

    # Find all filenames which contain cluster text
    # The items in this list are directory names
    #Each of which cotnain summaries of various sizees
    cluster_summs = [x for x in fetch_filenames('raw/2001/data/complete/') if not '/docs' in x and os.path.isdir(x)]
    cluster_summs.extend(['datasets/duc2001/data/complete/' + x for x in os.listdir('datasets/duc2001/data/complete') if
                          os.path.isdir('datasets/duc2001/data/complete/' + x) and os.path.exists(
                              'datasets/duc2001/data/complete/' + x + '/perdocs')])
    for file in cluster_summs:
        #Cluster id of the summary
        cluster = cid(file.split('/')[-1])
        # Mapping from wc to filepath
        summs = {50: file + '/50', 100: file + '/100', 200: file + '/200', 400: file + '/400'}

        #Iterate through all wc possibilities
        for size in summs.keys():
            total_path = summs[size]
            soup = BeautifulSoup(open(total_path), 'html.parser')
            clusters[cluster].summaries_by_len[size] = preprocess_text(soup.sum.text)
    return clusters



def convert_corpus_format():
    """Legacy method, used to convert an old corpus file format to the current format"""
    corpus_old_2001 = pickle.load(open('data/corpus.pickle', 'rb'))
    corpus_old_2004 = pickle.load(open('data/corpus2004.pickle', 'rb'))
    corpus_new_2001 = dict()
    corpus_new_2004 = dict()
    for k, v in corpus_old_2001.items():
        v_new = CorpusEntry2001(k)
        v_new.text = v[0]
        v_new.summaries = v[1]
        v_new.type = v[2]
        v_new.cluster = v[3]
        v_new.optimal = v[-2]
        v_new.optimal_labels = v[-1]

        corpus_new_2001[k] = v_new

    for k, v in corpus_old_2004.items():
        v_new = CorpusEntry2004(k)
        v_new.text = v[0]
        v_new.summaries = v[1]
        v_new.type = v[2]
        v_new.text_by_doc = v[3]

        v_new.optimal = v[-1]
        v_new.optimal_labels = [True if x in v_new.optimal else False for x in v_new.optimal]

        corpus_new_2004[k] = v_new
    pickle.dump(corpus_new_2001, open('data/corpus_new.pickle', 'wb'))

    pickle.dump(corpus_new_2004, open('data/corpus2004_new.pickle', 'wb'))

def get_sent_matrix(corpus, str2vec=str2vec_arora, type=1, oversample=True, diff=False):

    """
    Gets supervised training matrix formatted for scikit-learn

    :param corpus: Common corpus object
    :param type: Type of data to extract (training, dev, test, etc)
    :param diff: If true, subtracts document vectors from the
    :return: A tuple with two values
    [0]: A numpy matrix, with each row representing a document, and columns representing dimensions
    [1]: Labels for the documents. 1=summary, -1=sentences, in the same order as docs in [0]

    """
    vecs = []
    labels = []
    for i, v in enumerate(corpus.values()):
        if v.type == type:
            docvec = v.emb[str2vec.__name__][-1]
            # docvec = str2vec(' '.join(v.text))
            for i, sent in enumerate(v.text):
                vec = v.emb[str2vec.__name__][i] - docvec if diff else v.emb[str2vec.__name__][i]
                # vec = str2vec(sent) - docvec if diff else str2vec(sent)

                if oversample:
                    if v.optimal_labels[i]:
                        for _ in range(10):
                            labels.append(1)
                            vecs.append(vec.T)
                    else:
                        labels.append(0)
                        vecs.append(vec.T)
                else:
                    vecs.append(vec.T)
                    labels.append(1 if v.optimal_labels[i] else 0)


    data = np.stack(vecs, axis=0)
    return (data, np.asarray(labels))

import time


def add_extractive_summaries(corpus):
    """Applied find_summary to every document in the corpus"""
    for v in corpus.values():
        start = time.clock()
        v.append(find_summary(v, wl=100))
        duration = (time.clock() - start)
        print("{} seconds for {} sentences".format(duration, len(v.text)))


def find_summary(v, wl=100):
    """Greedily approximates optimal summaries by ROUGE scores"""
    import main
    pyrouge = main.pyrouge_obj()
    r = []
    total_words = 0

    while True:
        #Break if no sentences are left
        if len([x for x in v.text if not x in r]) == 0:
            break
        newSent = max([x for x in v.text if not x in r], key=functools.partial(rouge_with_summary, r, v.summaries))

        r.append(newSent)
        total_words += len(newSent.split(' '))
        if len(newSent.split(' ')) + total_words < wl:
            break
    print(pyrouge.score_summary(main.getPyrougeSum([r]), main.getPyrougeRef([v.summaries]))['rouge_2_f_score'])
    return r


ROUGE_path = "lib/RELEASE-1.5.5/ROUGE-1.5.5.pl"
data_path = "lib/RELEASE-1.5.5/data"

def rouge_with_summary(old, ref, new):
    """Computes the ROUGE score of a previous summary with a new sentence added
    :param old: Previously selected sentenes
    :param ref: The reference summary
    :param new: The new sentence to evaluate

    """
    import main
    pyrouge = main.pyrouge_obj()
    rouge = main.pythonrouge_obj()
    summary = old[:]
    summary.append(new)
    score = pyrouge.score_summary(main.getPyrougeSum([summary]), main.getPyrougeRef([ref]), force_hong=True)[
        'rouge_2_f_score']

    # setting_file = rouge.setting(files=False, summary=[summary], reference=[ref])
    #score = rouge.eval_rouge(setting_file, ROUGE_path=ROUGE_path, data_path=data_path)['ROUGE-2-F']
    return score


def label():
    """Converts a list of optimal summaries to a list of labels"""
    corpus = load2001()
    for v in corpus.values():
        v.optimal_labels = [True if x in v.optimal else False for x in v.optimal]
    pickle.dump(corpus, open('data/corpus_new.pickle', 'wb'))


def load2001():
    return pickle.load(open('data/corpus_new.pickle', 'rb'))


def load2004():
    return pickle.load(open('data/corpus2004_new.pickle', 'rb'))


def loadsemeval():
    return pickle.load(open('data/corpus_semeval.pickle', 'rb'))

def opex_stats():
    """Prints stats on optimal extractive summary"""
    import main
    pyrouge = main.pyrouge_obj()
    corpus = load2001()
    r = []
    for v in corpus.values():
        if v.type == 2:
            summary = [[x for i, x in enumerate(v.text) if v.optimal_labels[i]]]
            reference = [v.summaries]
            r.append(pyrouge.score_summary(main.getPyrougeSum(summary), main.getPyrougeRef(reference), force_hong=True))
    rouge2 = [x['rouge_1_recall'] for x in r]
    print(rouge2)
    print(sum(rouge2) / len(rouge2))


class SemEvalEntry:
    key = "INVALID_KEY"
    text = []
    candidates = []
    keywords = []

    def __init__(self, key):
        self.key = key

class CorpusEntry:
    """Represents a single document/cluster in a corpus"""
    key = "INVALID_KEY"
    text = []
    summaries = []
    type = -1

    emb = {}


    def __init__(self, key):
        self.key = key


class CorpusEntry2001(CorpusEntry):
    cluster = ""

    def __init__(self, key):
        self.key = key


class CorpusEntry2001Cluster(CorpusEntry):
    summaries_by_len = {50: [], 100: [], 200: [], 400: []}

    def __init__(self, key):
        self.key = key


class CorpusEntry2004(CorpusEntry):
    text_by_doc = []


    def __init__(self, key):
        self.key = key


class CorpusEntry2004(CorpusEntry):
    text_by_doc = []

    def __init__(self, key):
        self.key = key


from summarizer_modules import *


def normalize_vecs():
    """Retroactive legacy function to normalize skipthought embeddings"""
    corpus_2001 = pickle.load(open('data/corpus_new.pickle', 'rb'))
    corpus_2004 = pickle.load(open('data/corpus2004_new.pickle', 'rb'))
    for v in corpus_2001.values():
        for k, vecs in v.emb.items():
            v.emb[k] = [normalizeVec(vec) for vec in vecs]

    for v in corpus_2004.values():
        for k, vec in v.emb.items():
            v.emb[k] = [normalizeVec(vec) for vec in vecs]

    pickle.dump(corpus_2001, open('data/corpus_new.pickle', 'wb'))
    pickle.dump(corpus_2004, open('data/corpus2004_new.pickle', 'wb'))

def preprocess_emb(corpus):
    """Calculates and stores sentence embeddings to speed up computations"""

    # Fix bug with pickle pointer sharing
    str2vecs = [str2vec_avg, str2vec_arora, str2vec_doc2vec, str2vec_skipthought]
    print(len(corpus.items()))
    for k, v in corpus.items():
        if v.type == 2:
            v.emb={}
            for str2vec in str2vecs:
                r = [str2vec(s) for s in v.text]
                r.append(str2vec(' '.join(v.text)))
                v.emb[str2vec.__name__] = r
            print(v)
            print('emblen: {}'.format(len(v.emb)))
            corpus[k] = v
        print(k)
    return corpus

from summarizer_modules import str2vec_doc2vec
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "2004":
        pickle.dump(complete_data_2004(), open('data/corpus2004_new.pickle', 'wb'))
    elif len(sys.argv) > 1 and sys.argv[1] == "label":
        label()
    elif len(sys.argv) > 1 and sys.argv[1] == "matrix":
        matrix, labels = get_sent_matrix(load2001(), oversample=False, diff=True, str2vec=str2vec_doc2vec)
        matrix = np.insert(matrix, 0, labels, axis=1)
        np.savetxt('data/sentmatrix.txt', matrix)
    elif len(sys.argv) > 1 and sys.argv[1] == "opexstats":
        opex_stats()
    elif len(sys.argv) > 1 and sys.argv[1] == "convert":
        convert_corpus_format()
    elif len(sys.argv) > 1 and sys.argv[1] == "normalize":
        normalize_vecs()
    # elif len(sys.argv) > 1 and sys.argv[1] == "semeval":
    #    pickle.dump(complete_data_semeval(), open("data/corpus_semeval.pickle", 'wb'))
    elif len(sys.argv) > 1 and sys.argv[1] == "emb":
        corpus = None
        corpus2004 = None
        with open('data/corpus2004_new.pickle', 'rb') as f:
            corpus2004 = pickle.load(f)
            corpus2004 = preprocess_emb(corpus2004)
        with open('data/corpus2004_new.pickle', 'wb') as f:
            pickle.dump(corpus2004, f, protocol=4)

        with open('data/corpus_new.pickle', 'rb') as f:
            corpus = pickle.load(f)
            copus = preprocess_emb(corpus)
        with open('data/corpus_new.pickle', 'wb') as f:
            pickle.dump(corpus, f, protocol=4)
        pickle.dump(preprocess_emb(pickle.load(open('data/corpus2004_new.pickle', 'rb'))),
                    open('data/corpus2004_new.pickle', 'wb'), protocol=4)
        pickle.dump(preprocess_emb(pickle.load(open('data/corpus_new.pickle', 'rb'))),
                    open('data/corpus_new.pickle', 'wb'),
                    protocol=4)
    else:
        print("Running preprocessor")
        corpus = complete_data_2001()
        cluster_data(corpus)
        add_extractive_summaries(corpus)

        pickle.dump(corpus, open('data/corpus_new.pickle', "wb"))

        print(get_sent_matrix(corpus))
        print(get_sent_matrix(corpus))
