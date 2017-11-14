#!/usr/bin/python

import seaborn as sbn
import numpy as np
import argparse
import utils

from scipy.stats import pearsonr
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile", help="Input text file (corpus)", required=True)
    parser.add_argument("-c", "--cast", help="Cast file", required=True)
    parser.add_argument("-p", "--plotsave", help="Location to save heatmap image", default=None)
    parser.add_argument("-k", "--num_topics", help="Number of topics", type=int, default=5)
    parser.add_argument("-n", "--num_iterations", help="Number of LDA iterations", type=int, default=500)
    parser.add_argument("-t", "--topic_words", help="Number of top words per topic to print", type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")

    options = parser.parse_args()

    run(options.infile, options.cast, options.plotsave, options.num_topics, options.num_iterations, options.topic_words, options.verbose)


def run(infile, castfile, savefile=None, num_topics=5, num_iterations=500, num_topic_words=10, verbose=False):
    if verbose:
        print "Parsing files"
    sections = utils.parse_file_to_sections(infile)
    castdict = utils.parse_castfile_to_dict(castfile)
    characters = sorted(castdict.keys())
    if verbose:
        print "----Parsed"
        print "Cleaning sections"
    clean_sections_with_chars = utils.make_clean_sections(sections, None) # No cast removal

    if verbose:
        print "----Cleaned"
        print "Vectorizing data"
    data, vectorizer = utils.clean_and_vectorize(sections, castdict)

    if verbose:
        print "----Vectorized"
        print "Training LDA model"
    lda_model = utils.run_lda(data, num_topics, num_iterations)

    if verbose:
        print "----Trained"
        print "Calculating correlation"
    coeffs, pvals = calculate_correlations(clean_sections_with_chars, characters, lda_model)

    if verbose:
        print "----Calculated"

    
    utils.print_top_topic_words(lda_model, vectorizer.get_feature_names(), num_topic_words)

    if verbose:
        print "Plotting heatmap"
    plot_heatmap(coeffs, characters, savefile)

    return data, vectorizer, lda_model, characters, coeffs, pvals


def calculate_correlations(clean_sections, characters, lda_model):
    character_counts = utils.calculate_character_counts(clean_sections, characters)

    doc_topics = lda_model.doc_topic_

    pearson_coeffs = np.zeros((len(characters), doc_topics.shape[1]))
    pearson_pvals = np.zeros((len(characters), doc_topics.shape[1]))

    for topic_index in xrange(doc_topics.shape[1]):
        for character_index in xrange(len(characters)):
            ccs = character_counts[:,character_index]
            dts = doc_topics[:,topic_index]

            r, p = pearsonr(ccs, dts)
            pearson_coeffs[character_index, topic_index] = r
            pearson_pvals[character_index, topic_index] = p

    return pearson_coeffs, pearson_pvals



def plot_heatmap(correlations, characters, savefile=None):
    # Ref: https://plot.ly/python/heatmaps/
    topic_titles = ["Topic {}".format(i) for i in xrange(correlations.shape[1])]
    fig = plt.figure()
    r = sbn.heatmap(correlations, cmap="BuPu", xticklabels=topic_titles, yticklabels=characters)
    if savefile:
        plt.savefig(savefile)
    plt.show()



if __name__ == "__main__":
    main()







