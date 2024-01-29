"""
parsers to supper the textastic framework
"""
import json
from collections import Counter

def json_parser(filename):
    f = open(filename)
    raw = json.load(f)
    text = raw['text']
    words = text.split(" ")
    wc = Counter(words)
    num = len(wc)
    f.close()
    return {'wordcount':wc, 'numwords':num}