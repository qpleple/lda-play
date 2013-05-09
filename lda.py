#!/usr/bin/env python

# imports
import os, sys, logging
from termcolor import colored, cprint
from gensim import corpora, models, similarities, utils
from progressbar import *
from cPickle import load, dump
from utils import *

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

def bowCorpuses(paths):
    cprint("{0:.^80}".format(" Preprocessing "), 'yellow')
    for name, path in paths.items():
        print "Corpus", colored(name, 'green')
        
        save_to = path + '-output'
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        
        corpus_path = os.path.join(save_to, 'corpus.pickle')
        vocab_path  = os.path.join(save_to, 'vocab.pickle')
        if os.path.exists(corpus_path):
            print colored(corpus_path, 'green'), "already exists"
            continue

        corpus, vocab = bowCorpus(path)
        dump(corpus, open(corpus_path, 'w'))
        dump(vocab, open(vocab_path, 'w'))

def ldaCorpuses(paths):
    cprint("{0:.^80}".format(" LDA "), 'yellow')
    for name, path in paths.items():
        print "Corpus", colored(name, 'green')
        
        save_to  = path + '-output'
        lda_path = os.path.join(save_to, 'lda.pickle')

        if os.path.exists(lda_path):
            print colored(lda_path, 'green'), "already exists"
            continue

        corpus_path = os.path.join(save_to, 'corpus.pickle')
        vocab_path  = os.path.join(save_to, 'vocab.pickle')

        sz = bytesString(os.path.getsize(vocab_path))
        print "Loading vocab", colored(sz, 'green')
        vocab  = load(open(vocab_path, 'r'))
        sz = bytesString(os.path.getsize(corpus_path))
        print "Loading corpus", colored(sz, 'green')
        corpus = load(open(corpus_path, 'r'))

        lda = models.ldamodel.LdaModel(
            corpus       = corpus,
            id2word      = vocab,
            num_topics   = 50,
            update_every = 0,
            passes       = 1000
        )

        lda.save(lda_path)


# -- versions --

# 20news
# v1: 100 topics, 500 passes
# v2: 15 topics, 500 passes
# current: 50 topics, 1000 passes

# nips
# v1: 100 topics, 500 passes
# current: 50 topics, 1000 passes

paths = {
    '20news': '/Users/qt/research-data/20news/all',
    'nips'  : '/Users/qt/research-data/nips/all',
}
bowCorpuses(paths)
ldaCorpuses(paths)
