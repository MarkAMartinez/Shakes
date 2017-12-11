#!/usr/bin/python

import seaborn as sbn
import numpy as np
import argparse
import utils

from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input_file", help="Input file name (e.g. plays)", default="preprocessed_shakespeare.txt")
    parser.add_argument("-c", "--cast_file", help="File for cast list", default="aggregate_curated_cast.txt")
    parser.add_argument("-k", "--num_topics", help="Number of topics", type=int, default=20)
    parser.add_argument("-ni", "--num_iter", help="Number of iterations per training", type=int, default=1000)
    parser.add_argument("-n", "--num_trials", help="Number of times to train LDA model", type=int, default=10)
    parser.add_argument("-w", "--num_top_words", help="Number of top words to use", type=int, default=5)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")

    options = parser.parse_args()

    run(options.input_file, options.cast_file, options.num_topics, options.num_iter, options.num_trials, options.num_top_words, options.verbose)


def run(input_file, cast_file, num_topics, num_iter, num_trials, num_top_words, verbose):
    if verbose:
        print "Cleaning"
    sections = utils.parse_file_to_sections(input_file)
    castdict = utils.parse_castfile_to_dict(cast_file)

    if verbose:
        print "Vectorizing"
    data, vectorizer = utils.clean_and_vectorize(sections, castdict)

    top_word_dictsets = []

    for i in xrange(num_trials):
        if verbose:
            print "Run " + str(i+1) + " of " + str(num_trials) + "."
        lda_model = utils.run_lda(data, num_topics, num_iter)

        top_word_dictsets.append(utils.find_top_word_dictset(lda_model, vectorizer.get_feature_names(), num_top_words))

    jaccards = []

    if verbose:
        print "Calculating Jaccard scores"
    for first in top_word_dictsets:
        for second in top_word_dictsets:
            jaccards.append(utils.calc_jaccard_nocount(first, second))
    
    print "Average Jaccard between pairs: " + str(np.mean(jaccards))

    return np.mean(jaccards)















if __name__ == "__main__":
    main()






