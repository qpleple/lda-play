import os, sys, logging
from termcolor import colored, cprint
from gensim import corpora, models, similarities, utils
from progressbar import *
from cPickle import load, dump

logging.basicConfig(format = '%(levelname)s: %(message)s', level = logging.INFO)

widgets = [SimpleProgress(), ' ', Bar(marker='-',left='[',right=']'), ' ',
    ETA(), ' (at', FileTransferSpeed(unit='it'), ')', ]
pbar = ProgressBar(widgets = widgets)

     
def bowCorpus(root_path):
    vocab = corpora.dictionary.Dictionary()
    corpus = []
    filenames = [os.path.join(root_path, f) for f in os.listdir(root_path)]
    
    print colored(len(filenames), 'green'), "files found in", colored(root_path, 'green')
    
    print "Converting each file into bag-of-word:"
    for fname in pbar(filenames):
        with open(fname, 'r') as f:
            content = f.read()

        tokens = utils.lemmatize(content)
        # lemmatize return strings like 'moderate/VB' or 'listing/NN'
        tokens = [x.split('/')[0] for x in tokens]
        bow = vocab.doc2bow(tokens, allow_update = True)
        corpus.append(bow)

    return corpus, vocab

corpus = {'20news': '/Users/qt/Desktop/research-data/20news/all'}
for name, path in corpus.items():
    print "Corpus", colored(name, 'green')
    save_to = path + '-output'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    corpus_path = os.path.join(save_to, 'corpus.pickle')
    vocab_path  = os.path.join(save_to, 'corpus.vocab')
    if os.path.exists(corpus_path):
        print colored(corpus_path, 'green'), "already exists"
        continue
    corpus, vocab = bowCorpus(path)
    dump(corpus, open(corpus_path, 'w'))
    dump(vocab, open(vocab_path, 'w'))
    