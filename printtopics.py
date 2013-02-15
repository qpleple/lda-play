import sys, os
from termcolor import colored, cprint
from utils import *

path = sys.argv[1]
sz = bytesString(os.path.getsize(path))
print "Loading model", colored(sz, 'green')
with open(path, 'r') as f:
    model = load(f)
top_words = model.show_topics(-1, 20, False, False)

for topic, words in enumerate(top_words):
    print colored(topic, 'yellow'), ' '.join([w for (p, w) in words if len(w) > 2])