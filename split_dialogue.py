#!/usr/bin/python

import nltk
import numpy as np
import argparse
import string
import utils

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="Input text filename", required=True)
    parser.add_argument("-c", "--cast", help="Cast text file", required=True)
    parser.add_argument("-o", "--output", help="Output filename", default=None)

    options = parser.parse_args()

    process(options.input, options.cast, options.output)




def process(corpus_filename, cast_filename, output=None):
    remove = string.punctuation
    # stripped_corpus = open(corpus_filename).read().translate(None, remove).lower()
    corpus = open(corpus_filename).read().lower()
    stripped_corpus = corpus.translate(None, remove)

    stops = stopwords.words("english")
    stopdict = dict((s.lower(),None) for s in stops)

    cast_dict = {}
    for line in open(cast_filename):
        words = nltk.word_tokenize(line.strip())
        for word in words:
            if (word.lower() not in stopdict):
                cast_dict[word.lower()] = True

    # cast_dialogue = {}
    # temp_text = ""
    # first = True

    # for word in stripped_corpus.split():
    #     if (word in cast_dict):
    #         if (not first):
    #             old_text = cast_dialogue[curr_char]

    cast_dialogue = {}
    prev_cast_member = None
    to_add = ""

    for word in corpus.split():
        if (word in cast_dict):
            if prev_cast_member is not None:
                if not prev_cast_member in cast_dialogue:
                    cast_dialogue[prev_cast_member] = ""
                cast_dialogue[prev_cast_member] += to_add
                to_add = ""
            prev_cast_member = word
        else:
            to_add += " " + word

    # for line in corpus.split('\n'):
    #     cast_split = line.strip().split('.')
    #     if (len(cast_split) <= 1):
    #         continue
    #     cast_member = cast_split[0].translate(None, (remove + " ")).lower()

    #     if not cast_member in cast_dict:


    #     cast_line = ".".join(cast_split[1:])
        
    #     if not cast_member in cast_dialogue:
    #         cast_dialogue[cast_member] = ""

    #     cast_dialogue[cast_member] += " " + " ".join(nltk.word_tokenize(cast_line.strip()))


    data, vectorizer = utils.clean_and_vectorize(cast_dialogue.values(), cast_dict)
    vocab = vectorizer.get_feature_names()

    print cast_dict.keys()

    # print vocab


    return data, vectorizer, vocab, cast_dict





if __name__ == "__main__":
    main()