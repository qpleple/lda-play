import sys, os
from termcolor import colored, cprint
from utils import *

path = sys.argv[1]
sz = bytesString(os.path.getsize(path))
print "Loading model", colored(sz, 'green')
with open(path, 'r') as f:
    model = load(f)

top_words = model.show_topics(-1, 30, False, False)

classif = {}
for topic, words in enumerate(top_words):
    cprint("Topic {}".format(topic), 'yellow')
    words = [w for (p, w) in words if len(w) > 2]
    print "\n".join(words[:10])
    print " ".join(words[10:])
    print