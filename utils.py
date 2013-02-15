import math
from cPickle import load, dump

def bytesString(bytes):
    power = int(math.log(bytes, 1000))
    scaled = bytes / 1000.**power
    prefixes = ' kMGT'
    return '%.1f %sB' % (scaled, prefixes[power])
