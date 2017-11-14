#!/usr/bin/python

import lda
import nltk
import string
import numpy as np
import parse_corpus

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def parse_to_vectorized(infile, castfile):
    castdict = parse_castfile_to_dict(castfile)
    sections = parse_file_to_sections(infile)
    data, vectorizer = clean_and_vectorize(sections, castdict)

    return data, vectorizer


def parse_file_to_sections(infile):
    sections = parse_corpus.process(infile)
    return sections


def parse_castfile_to_dict(castfile):
    stopdict = dict((s.lower(), None) for s in stopwords.words("english"))
    castdict = {}
    for line in open(castfile):
        words = nltk.word_tokenize(line.strip())
        for word in words:
            if word.lower() not in stopdict:
                castdict[word.lower()] = True

    return castdict



def make_clean_sections(sections, castdict=None):
    stops = stopwords.words("english")
    stopdict = dict((s.lower(),None) for s in stops) # Sets are really terrible in Python
    # print [s for s in stopdict.iterkeys()]

    if castdict is None:
        castdict = {}

    ps = PorterStemmer()

    clean_sections = []

    for section in sections:
    #     secwords = section.split()
    #     tokens = nltk.word_tokenize(" ".join(secwords))
        tokens = nltk.word_tokenize(section)
        nonstops = [w for w in tokens if not (w.lower() in stopdict or w.lower() in castdict)]
        stemmed = [ps.stem(t.lower()) for t in nonstops if t.isalnum()]
    #     nonstops = [w for w in stemmed if not (w in stopdict or w in castdict)]
        clean_sections.append(" ".join(stemmed)) # Note this does not preserve structure,
                                                  #      but all words are now present in the section string


    return clean_sections



def clean_and_vectorize(sections, castdict=None):
    # print sections
    # print "!!!!!!!!!"
    clean_sections = make_clean_sections(sections, castdict)

    vectorizer = CountVectorizer(analyzer = "word")

    data = vectorizer.fit_transform(clean_sections)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    # train_data_features = data.toarray()

    # vocab = vectorizer.get_feature_names()

    return data, vectorizer


def run_lda(data, num_topics=5, num_iter=500):
    model = lda.LDA(n_topics=num_topics, n_iter=num_iter)
    model.fit(data)

    return model


# Will be all 0 if clean_sections had cast members removed
def calculate_character_counts(clean_sections, chars):
    counts = np.zeros((len(clean_sections), len(chars)))
    for (i, section) in enumerate(clean_sections):
        for (j, char) in enumerate(chars):
            counts[i,j] = section.count(char)

    return counts












