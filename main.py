import ducpreprocess
import rouge as myrouge
import summarizer
from ducpreprocess import *
from pyrouge import Rouge155
from pythonrouge import pythonrouge
from summarizer_modules import *

# from pythonrouge import pythonrouge

pyrouge = Rouge155(rouge_home="lib/RELEASE-1.5.5", n_words=100, stem=True)

# Configure rouge
ROUGE_path = "lib/RELEASE-1.5.5/ROUGE-1.5.5.pl"
data_path = "lib/RELEASE-1.5.5/data"
rouge = pythonrouge.Pythonrouge(n_gram=4, ROUGE_SU4=False, ROUGE_L=False, stemming=True, stopwords=False,
                                word_level=True,
                                length_limit=True, length=100, use_cf=True, cf=95, scoring_formula="average",
                                resampling=True,
                                samples=1000, favor=True, p=.5)

options = "-n 4 -m -a -l 100 -x -c 95 -r 1000 -f A -p 0.5 -t 0".split(' ')

# Preprocessing of data
#Find lengths of summaries, for a quick monitoring tool
sum_len = []


def pyrouge_obj():
    return pyrouge


def pythonrouge_obj():
    return rouge


def getPyrougeRef(references):
    r = dict()
    references = references[0]
    for i, ref in enumerate(references):
        r[str(chr(ord('A') + i))] = [r.split(" ") for r in ref]
    return r


def getPyrougeSum(summary):
    return [r.split(" ") for r in summary[0]]



def run(str2vec, selector, args, type):
    corpus = None
    if type == "2004":
        corpus = ducpreprocess.load2004()
    else:
        corpus = ducpreprocess.load2001()
    results = []
    results2 = []
    results3 = []

    if type == "2004":
        for v in corpus.values():
            # If doc is devtest
            if v.type == 3:
                args['docs'] = v.text_by_doc
                summary = [
                    summarizer.summarize(v, str2vec, selector, 100, args)]
                # summary2 = [summarizer.summarize(v.text, str2vec, selector_near, 100, args)]
                # if summary != summary2:
                #    print(summary)
                #    print(summary2)
                # Allow easy debug step-through
                assert len(' '.join(summary[0]).split()) <= 100
                sum_len.append(len(summary[0]))
                reference = [v.summaries]

                setting_file = rouge.setting(files=False, summary=summary, reference=reference)
                results.append(rouge.eval_rouge(setting_file, ROUGE_path=ROUGE_path, data_path=data_path))
                results2.append(
                    pyrouge.score_summary(getPyrougeSum(summary), getPyrougeRef(reference), force_hong=True))
                results3.append(myrouge.compute_rouge(' '.join(summary[0]), [' '.join(x) for x in reference[0]]))

    elif type == "single":
        for v in corpus.values():
            # If doc is devtest
            if v.type == 1:
                # Create summary in a modular way
                # Note the dummy list to match pythonrouge's format
                summary = [
                    summarizer.summarize(v, str2vec, selector, 100, args=args)]
                sum_len.append(len(summary[0]))
                # Extract reference summary
                reference = [v.summaries]
                # if random.random() < .03:
                # print("Ex")
                # print(summary)
                # print(reference)

                setting_file = rouge.setting(files=False, summary=summary, reference=reference)
                results.append(rouge.eval_rouge(setting_file, ROUGE_path=ROUGE_path, data_path=data_path))
                results2.append(
                    pyrouge.score_summary(getPyrougeSum(summary), getPyrougeRef(reference), force_hong=True))


    elif type == "cluster":
        clusters = ducpreprocess.cluster_data(corpus)
        for v in clusters.values():
            # If doc is devtest
            if v.type == 2:
                summary = [
                    summarizer.summarize(v, str2vec, selector, 100, args=args)]
                sum_len.append(len(summary[0]))
                reference = [[v.summaries_by_len[100]]]

                setting_file = rouge.setting(files=False, summary=summary, reference=reference)
                results.append(rouge.eval_rouge(setting_file, ROUGE_path=ROUGE_path, data_path=data_path))
                results2.append(pyrouge.score_summary(getPyrougeSum(summary), getPyrougeRef(reference), force_hong=True))

    print("Rouge 1: {}".format(sum([r['ROUGE-1-R'] for r in results]) * 1.0 / len(results)))
    print("Rouge 2: {}".format(sum([r['ROUGE-2-R'] for r in results]) * 1.0 / len(results)))
    rouge1 = sum([r['rouge_1_recall'] for r in results2]) * 1.0 / len(results2)
    rouge2 = sum([r['rouge_2_recall'] for r in results2]) * 1.0 / len(results2)
    rouge4 = sum([r['rouge_4_recall'] for r in results2]) * 1.0 / len(results2)

    print("Rouge 1 (pyrouge): {}".format(rouge1))
    print("Rouge 2 (pyrouge): {}".format(rouge2))

    print("Rouge 4 (pyrouge): {}".format(rouge4))
    print(sum_len)
    print([r['rouge_1_recall'] for r in results2])

    # print("Rouge 1 (me): {}".format(np.mean([x[0] for x in results3])))
    # print("Rouge 2 (me): {}".format(np.mean([x[1] for x in results3])))

    return rouge1, rouge2, [r['rouge_1_recall'] for r in results2]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "semeval":
        corpus = ducpreprocess.loadsemeval()
        for v in corpus.values():
            r = summarizer.keywords(v, str2vec_arora, selector_greedy, 10, {})
            real = v.keywords[:10]
            intersection = len([x for x in r if x in real])
            print(intersection)
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        selectors = [selector_greedy, selector_near, selector_lexrank_continuous, selector_lexrank_discrete,
                     selector_max_sim,
                     selector_near_then_bf, selector_near_redundancy,
                     selector_optimize_against_redundancy, selector_cluster, selector_svm,
                     selector_pca, selector_pca_2, selector_lead_baseline,
                     selector_near_adjusted, selector_far, selector_combined, selector_cluster,
                     selector_random]
        cap_flags = [False]
        str2vecs = [str2vec_avg, str2vec_arora, str2vec_doc2vec, str2vec_skipthought]
        # types = ["single", "2004"]
        types = ["2004"]
        if os.path.isfile('data/results.pickle'):
            results = pickle.load(open('data/results.pickle', 'rb'))
        else:
            results = dict()
        print(results)

        for params in itertools.product(selectors, cap_flags, str2vecs, types):
            if not params in results:
                orig_selector, cap_flag, str2vec, type = params
                selector = orig_selector
                if cap_flag and orig_selector != selector_random:
                    selector = make_capped(orig_selector)
                args = {}
                if selector == selector_svm and str2vec == str2vec_doc2vec:
                    args['svm'] = 'doc2vec'
                if (
                            selector == selector_svm or selector == selector_greedy or selector == selector_max_sim) and str2vec == str2vec_skipthought:
                    continue
                # functools.partial does not preserve __name__
                print("Results for {}-{} ({})".format(str2vec.__name__, orig_selector.__name__, type))
                results[params] = run(str2vec, selector, args, type)

            # matrix[selectors.index(orig_selector)][str2vecs.index(str2vec)] = "{}({})".format(rouge1, rouge2)

            pickle.dump(results, open('data/results.pickle', 'wb'))

    else:
        run(str2vec_arora, selector_near,
            {'docvec_avg': False, 'greedy_avg': False, 'pca': False, 'svm': 'doc2vec', 'skip_cache': False},
            "2004")
        from summarizer_modules import *
"""
List of args:
pca: Perform pca on vectors before passing to selector
docvec_avg: Make docvec by averaging together vectors instead of calling str2vec
greedy_avg: Greedy selctor optimizes the average of vectors, not the result of str2vec
"""
