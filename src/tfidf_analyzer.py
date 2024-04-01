import json
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


corpus = json.load(open("./sharegpt_data/sharegpt.text.json", "r"))
random.shuffle(corpus)
corpus = corpus[:10000]
corpus = [line.replace("Human: ","") for line in corpus]
corpus = [line.replace("Assistant: ", "\n\n") for line in corpus]
corpus = [line.replace("<|end_of_turn|>","") for line in corpus]

# corpus = ['This is the first document.',
#           'This document is the second document.',
#           'And this is the third one.',
#           'Is this the first document?']
def analyze_ngrams(corpus, n=2):
    # Getting trigrams 
    stop_words = stopwords.words('english')
    # stop_words.extend(["sentence", "following", "name", "list", "the", "given", "example", "it", "also", "make"])
    stop_words = set(stop_words)
    for i, line in enumerate(corpus):
        corpus[i] = ' '.join([x for x in word_tokenize(line) if x not in stop_words])
    vectorizer = CountVectorizer(ngram_range = (n,n))
    X1 = vectorizer.fit_transform(corpus) 
    features = (vectorizer.get_feature_names_out())

    # Applying TFIDF
    vectorizer = TfidfVectorizer(ngram_range = (n,n))
    X2 = vectorizer.fit_transform(corpus)
    scores = (X2.toarray())
    
    # Getting top ranking features
    sums = X2.sum(axis = 0)
    data1 = []
    for col, term in enumerate(features):
        data1.append( (term, sums[0, col] ))
    ranking = pd.DataFrame(data1, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    print ("\n\nWords head : \n", words.head(10))

def analyze_unigram(corpus):
    vectorizer = TfidfVectorizer(stop_words='english',)
    tfidf_result = vectorizer.fit_transform(corpus)
    scores = zip(vectorizer.get_feature_names_out(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores[:20]:
        print("{0:10} Score: {1}".format(item[0], item[1]))

# analyze_ngrams(corpus, n=2)
analyze_unigram(corpus)