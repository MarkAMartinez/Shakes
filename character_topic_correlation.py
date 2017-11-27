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

    parser.add_argument("-i", "--infile", help="Input text file (corpus)", required=True)
    parser.add_argument("-c", "--cast", help="Cast file", required=True)
    parser.add_argument("-p", "--plotsave", help="Directory in which to save heatmap and topic word distributions", default=None)
    parser.add_argument("-k", "--num_topics", help="Number of topics", type=int, default=5)
    parser.add_argument("-n", "--num_iterations", help="Number of LDA iterations", type=int, default=500)
    parser.add_argument("-t", "--topic_words", help="Number of top words per topic to print", type=int, default=10)
    parser.add_argument("-r", "--correlation_threshold", help="Portion of restults to plot in heatmap", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")

    options = parser.parse_args()

    run(options.infile, options.cast, options.plotsave, options.num_topics, options.num_iterations, options.topic_words, options.correlation_threshold, options.verbose)


def run(infile, castfile, savepath=None, num_topics=5, num_iterations=500, num_topic_words=10, restrict_portion=0.1, verbose=False):
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
    coeffs, pvals, bh_fdrs = calculate_correlations(clean_sections_with_chars, characters, lda_model)
    annotations = generate_heatmap_annotations(coeffs, bh_fdrs)


    if verbose:
        print "----Calculated"

    if savepath:
        utils.save_topic_distributions(vectorizer, lda_model, num_topic_words, savepath)
    utils.print_top_topic_words(lda_model, vectorizer.get_feature_names(), num_topic_words)

    if verbose:
        print "Plotting heatmap"
    plot_heatmap(coeffs, bh_fdrs, characters, restrict_portion, savepath)

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

    bh_fdrs = correct_bh_fdr(pearson_pvals)

    return pearson_coeffs, pearson_pvals, bh_fdrs



def plot_heatmap(correlations, bh_fdrs, characters, threshold=0.1, savepath=None):
    # Ref: https://plot.ly/python/heatmaps/
    # print np.array(annotations)
    topic_indices = range(correlations.shape[1])

    if threshold:
        mask = calculate_accepted_values(correlations, threshold)
        row_accept = [i for i in xrange(mask.shape[0]) if mask[i].any()]
        col_accept = [i for i in xrange(mask.shape[1]) if mask[:,i].any()]
        # row_accept = [mask[i].any() for i in xrange(mask.shape[0])]
        # col_accept = [mask[:,i].any() for i in xrange(mask.shape[1])]
        correlations = correlations[row_accept][:,col_accept]
        # print len(row_accept)
        # print len(col_accept)
        # print correlations
        # print correlations.shape
        characters = np.array(characters)[row_accept]
        topic_indices = np.array(topic_indices)[col_accept]
        bh_fdrs = bh_fdrs[row_accept][:,col_accept]

    annotations = generate_heatmap_annotations(correlations, bh_fdrs)

    topic_titles = ["Topic {}".format(i) for i in topic_indices]
    fig = plt.figure()
    # r = sbn.heatmap(correlations, cmap="BuPu", xticklabels=topic_titles, yticklabels=characters, annot=np.array(annotations), fmt="s")
    r = sbn.heatmap(correlations, cmap="BuPu", xticklabels=topic_titles, yticklabels=characters)
    if savepath:
        plt.savefig("{}/character_correlation_heatmap.pdf".format(savepath))
    plt.show()


def correct_bh_fdr(pvals, alpha=0.05):
    pval_array = []
    for row in pvals:
        for itm in row:
            pval_array.append(itm)
    
    bh_results = multipletests(pval_array, method="fdr_bh")
    reject_array = bh_results[0]
    corrected_array = bh_results[1]
    # print corrected_array
    # print len(corrected_array)
    corrected = np.zeros(pvals.shape)
    reject = np.zeros(pvals.shape, dtype=bool)

    for i in xrange(pvals.shape[0]):
        for j in xrange(pvals.shape[1]):
            corrected[i,j] = corrected_array[(i*pvals.shape[1])+j]
            # reject[i,j] = reject_array[(i*pvals.shape[0])+j]

    # reject, corrected = multipletests(pvals, method="fdr_bh")

    return corrected



def generate_heatmap_annotations(coefficients, pvals):
    # annotations = np.empty((coefficients.shape[0], coefficients.shape[1]), dtype=str)
    annotations = [["" for j in xrange(coefficients.shape[1])] for i in xrange(coefficients.shape[0])]
    for i in xrange(coefficients.shape[0]):
        for j in xrange(coefficients.shape[1]):
            # new_annot = ""
            # new_annot += "{0:.4f}".format(coefficients[i,j])
            # new_annot += ", "
            # new_annot += "{0:.4f}".format(pvals[i,j])
            annotations[i][j] = "{0:.3f}, {1:.3f}".format(coefficients[i,j], pvals[i,j])
            # annotations[i][j] = new_annot

    return annotations


def calculate_accepted_values(values, cutoff):
    # print "!!!!!!!!!!!"
    # print cutoff
    row_maxes = np.max(values, axis=1)
    row_maxes[np.isnan(row_maxes)] = 0
    # print row_maxes
    # svals = sorted(values.flatten(), key=lambda x: abs(x))
    svals = sorted(row_maxes.flatten(), key=lambda x: abs(x))
    # TODO There are are some serious bugs here it seems...
    #      Any potential nan issues here? NaNs are definitely showing up later on.
    #      Even sorted doesn't seem to be working right though...
    # print svals
    # print len(svals)
    if cutoff < 1.0:
        cutoff_val = svals[int(len(svals) * (1-cutoff))]
    else:
        cutoff_val = svals[len(svals) - int(cutoff)]
    # print cutoff_val

    testvals = np.array(values)
    testvals[np.isnan(testvals)] = 0

    mask = np.abs(testvals) >= cutoff_val
    # print mask.all()
    # print "!!!!!!!!!!!!!!!"
    # row_accept = [mask[i].any() for i in xrange(mask.shape[0])]
    # col_accept = [mask[:,i].any() for i in xrange(mask.shape[1])]

    # return row_accept, col_accept
    return mask




if __name__ == "__main__":
    main()







