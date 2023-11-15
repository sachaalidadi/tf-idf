import numpy as np
import pandas as pd
import nltk
import re
from collections import Counter
import sys
# nltk.download('punkt')
# nltk.download('stopwords')

def encode(text):
    # stopwords is a list of common words that do not add value to the sentence
    stopwords = nltk.corpus.stopwords.words('english')
    # We remove all non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    # We convert all characters to lowercase
    text = text.lower()
    # We remove all the stopwords
    text = text.strip()
    # We tokenize the text
    tokens = nltk.tokenize.word_tokenize(text)
    # If the token is not a common word, we ignore it
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

def tf(text):
    # tf compute the term frequency of each token in the text
    # We use our function encode to tokenize the text
    tokens = encode(text)
    # We use the Counter function to count the number of times each token appears in the text
    tf = Counter(tokens)
    for i in tf:
        # for each token, we divide the number of times it appears by the total number of tokens
        tf[i] = tf[i]/float(len(tokens))
    return tf

def idf(text):
    # idf compute the inverse document frequency of each token in the text
    tokens = encode(text)
    idf = {}
    for token in tokens:
        # for each token, we count the number of documents in which it appears
        idf[token] = idf.get(token, 0) + 1
    for i in idf:
        # for each token, we divide the number of documents by the number of documents in which it appears
        idf[i] = np.log(len(tokens)/float(idf[i]))
    return idf

def tfidf(text):
    # tfidf compute the product of tf and idf
    tfidf = {}
    _tf = tf(text)
    _idf = idf(text)
    for i in _tf:
        tfidf[i] = _tf[i] * _idf[i]
        # We normalize the tfidf vector
        tfidf[i] /= np.linalg.norm(list(tfidf.values()))
    return tfidf

# Example
# if argv is empty, use default text
if len(sys.argv) == 1:
    d = "The quick brown fox jumped over the lazy dog. The dog slept over the verandah."
else:
    d = open(sys.argv[1]).read()
print(d)
d = tfidf(d)
print(d)