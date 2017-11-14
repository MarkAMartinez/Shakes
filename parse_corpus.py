#!/usr/bin/python

import nltk
import numpy
import argparse

from sklearn.feature_extraction.text import CountVectorizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="Input filename", required=True)
    parser.add_argument("-o", "--output", help="Output filename", required=True)

    options = parser.parse_args()

    sections = process(options.input)

    write_sections(sections)


def process(filename):
    infile = open(filename)

    sections = []

    newsec = []

    for line in infile:
        if line.startswith("--------"):
            to_add = clean_section(newsec)
            if len(to_add) > 0:
                sections.append(to_add)
            newsec = []
        else:
            newsec.append(line)

    infile.close()

    return sections

    # Ref: http://scikit-learn.org/stable/modules/feature_extraction.html
    # vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)

    # vectorized = vectorizer.fit_transform(sections)

    # return vectorized


def write_sections(sections):
    pass


def clean_section(seclines):
    while len(seclines) > 1 and seclines[0].strip() == "":
        seclines = seclines[1:]

    if len(seclines) == 0:
        return []

    if seclines[0].startswith("<<THIS ELECTRONIC VERSION"):
        return []

    if not (seclines[0].strip().startswith("ACT ") or seclines[0].strip().startswith("SCENE ") or seclines[0].strip().startswith("ACT_")):
        return []

    while ((len(seclines) > 0) and (seclines[0].strip() == "" or seclines[0].strip().startswith("ACT") or seclines[0].strip().startswith("SCENE"))):
        seclines = seclines[1:]

    if len(seclines) < 1:
        return []

    return "\n".join(seclines)

















if __name__ == "__main__":
    main()




















