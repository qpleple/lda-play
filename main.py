import os, sys, logging
from termcolor import colored, cprint
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities

logging.basicConfig(format = '%(levelname)s: %(message)s', level = logging.INFO)
     
def bowCorpus(root_path):
    save_to = root_path + '-words'

    filenames = [os.path.join(root_path, f) for f in os.listdir(root_path)]
    print colored(len(filenames), 'green'), "files found in", colored(root_path, 'green')
    
    cv = CountVectorizer(
        input         = 'filename',
        charset_error = 'ignore',
        strip_accents = 'ascii',
        stop_words    = 'english'
    )

    print "Computing bag of word representation"
    counts = cv.fit_transform(filenames)
    print "Vocabulary of", colored(len(cv.vocabulary_), 'green'), "terms"

    print "Convert into a lists of tuples"
    # [[]] * n would put the same instance of [] n times
    corpus = [[] for _ in range(max(counts.row) + 1)]
    for (r, c, d) in zip(counts.row, counts.col, counts.data):
        corpus[r].append((c, d))

    return corpus



corpus = {'20news': '/Users/qt/Desktop/research-data/20news/all'}
for name, path in corpus.items():
    print "Corpus", colored(name, 'green')
    corpus = bowCorpus(path)
    tfidf = models.TfidfModel(corpus)


# http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit
# http://scikit-learn.github.com/scikit-learn-tutorial/working_with_text_data.html
# http://radimrehurek.com/gensim/tutorial.html#id2
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix