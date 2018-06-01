# -*- coding: utf-8 -*-
import argparse
import codecs, itertools, msgpack, re, sys, os, json
from collections import Counter
from functools import partial
from math import log
import cPickle as pickle
from random import shuffle
import numpy as np
from scipy import sparse

class ConstructNetwork:
    """ This program aims to construct co-occurrence network """

    def __init__(self, src_folder="../data/", min_count=5, window_size=2,\
            vocab_path=None, cooccur_path=None, auto_window=False, verbose=False):

        # initialization
        self.src = os.path.join(src_folder, "corpus/starred_reviews.txt")
        self.src_candidates = os.path.join(src_folder, "lexicon/candidates.json")
        self.filename = "starred.txt"
        self.dst_dir = os.path.join(src_folder, "network/")
        self.dst_voc_dir = os.path.join(self.dst_dir, "vocab/")
        self.dst_coo_dir = os.path.join(self.dst_dir, "cooccur/")
        self.dst_coo_path = os.path.join(self.dst_coo_dir, self.filename)
        self.verbose = verbose

        self.corpus = codecs.open(self.src, "r", encoding="utf-8")
        self.min_count = min_count
        self.window_size = window_size
        self.vocab_path = vocab_path
        self.cooccur_path = cooccur_path
        self.auto_window = auto_window


    def get_or_build(self, path, build_fn, *args, **kwargs):
        """ Load from serialized form or build an object, saving the built object.
            Remaining arguments are provided to `build_fn`.
        """

        save = False
        obj = None

        if path is not None and os.path.isfile(path):
            with open(path, "rb") as obj_f:
                obj = msgpack.load(obj_f, use_list=False, encoding="utf-8")
        else:
            save = True

        if obj is None:
            obj = build_fn(*args, **kwargs)

            if save and path is not None:
                with open(path, "wb") as obj_f:
                    msgpack.dump(obj, obj_f)

        return obj


    def build_vocab(self, corpus):
        """
        Build a vocabulary with word frequencies for an entire corpus.

        Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
        word ID and word corpus frequency.
        """

        if self.verbose:
            print "Building vocab from corpus"

        vocab = Counter()
        for line in corpus:
            tokens = line.strip().split()
            vocab.update(tokens)

        if self.verbose:
            print "Done building vocab from corpus."

        return {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}


    def build_cooccur(self, vocab, corpus, window_size=5, min_count=None):
        """
        Build a word co-occurrence list for the given corpus.

        This function is a tuple generator, where each element (representing
        a cooccurrence pair) is of the form

            (i_main, i_context, cooccurrence)

        where `i_main` is the ID of the main word in the cooccurrence and
        `i_context` is the ID of the context word, and `cooccurrence` is the
        `X_{ij}` cooccurrence value as described in Pennington et al.
        (2014).

        If `min_count` is not `None`, cooccurrence pairs where either word
        occurs in the corpus fewer than `min_count` times are ignored.
        """

        vocab_size = len(vocab)
        id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

        # Collect cooccurrences internally as a sparse matrix for passable
        # indexing speed; we'll convert into a list later
        cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                          dtype=np.float64)

        if self.verbose:
            print "Building cooccurrence matrix"

        for i, line in enumerate(corpus):
            tokens = line.strip().split()
            token_ids = [vocab[word][0] for word in tokens]

            for center_i, center_id in enumerate(token_ids):
                # Collect all word IDs in left window of center word
                context_ids = token_ids[max(0, center_i - window_size) : center_i]
                contexts_len = len(context_ids)

                for left_i, left_id in enumerate(context_ids):
                    # Distance from center word
                    distance = contexts_len - left_i

                    # Weight by inverse of distance between words
                    #increment = 1.0 / float(distance)
                    increment = 1.0

                    # Build co-occurrence matrix symmetrically (pretend we
                    # are calculating right contexts as well)
                    cooccurrences[center_id, left_id] += increment
                    cooccurrences[left_id, center_id] += increment
        # Now yield our tuple sequence (dig into the LiL-matrix internals to
        # quickly iterate through all nonzero cells)
        for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,
                                                       cooccurrences.data)):
            if min_count is not None and vocab[id2word[i]][1] < min_count:
                continue

            for data_idx, j in enumerate(row):
                if min_count is not None and vocab[id2word[j]][1] < min_count:
                    continue

                yield i, j, data[data_idx]


    def run_iter(self, vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
        """
        Run a single iteration of GloVe training using the given
        cooccurrence data and the previously computed weight vectors /
        biases and accompanying gradient histories.

        `data` is a pre-fetched data / weights list where each element is of
        the form

            (v_main, v_context,
             b_main, b_context,
             gradsq_W_main, gradsq_W_context,
             gradsq_b_main, gradsq_b_context,
             cooccurrence)

        as produced by the `train_glove` function. Each element in this
        tuple is an `ndarray` view into the data structure which contains
        it.

        See the `train_glove` function for information on the shapes of `W`,
        `biases`, `gradient_squared`, `gradient_squared_biases` and how they
        should be initialized.

        The parameters `x_max`, `alpha` define our weighting function when
        computing the cost for two word pairs; see the GloVe paper for more
        details.

        Returns the cost associated with the given weight assignments and
        updates the weights by online AdaGrad in place.
        """

        global_cost = 0

        # We want to iterate over data randomly so as not to unintentionally
        # bias the word vector contents
        shuffle(data)

        for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
             gradsq_b_main, gradsq_b_context, cooccurrence) in data:

            weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

            # Compute inner component of cost function, which is used in
            # both overall cost calculation and in gradient calculation
            cost_inner = (v_main.dot(v_context)
                          + b_main[0] + b_context[0]
                          - log(cooccurrence))

            # Compute cost
            cost = weight * (cost_inner ** 2)

            # Add weighted cost to the global cost tracker
            global_cost += 0.5 * cost

            # Compute gradients for word vector terms.
            # NB: `main_word` is only a view into `W` (not a copy), so our
            # modifications here will affect the global weight matrix;
            # likewise for context_word, biases, etc.
            grad_main = weight * cost_inner * v_context
            grad_context = weight * cost_inner * v_main

            # Compute gradients for bias terms
            grad_bias_main = weight * cost_inner
            grad_bias_context = weight * cost_inner

            # Now perform adaptive updates
            v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
            v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

            b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
            b_context -= (learning_rate * grad_bias_context / np.sqrt(
                    gradsq_b_context))

            # Update squared gradient sums
            gradsq_W_main += np.square(grad_main)
            gradsq_W_context += np.square(grad_context)
            gradsq_b_main += grad_bias_main ** 2
            gradsq_b_context += grad_bias_context ** 2

        return global_cost


    def create_dirs(self):
        """ create the directory if not exist"""

        if not os.path.exists(self.dst_dir):
            if self.verbose:
                print "Creating directory: " + "\033[1m" + self.dst_dir + "\033[0m"
            os.makedirs(self.dst_dir)
        if not os.path.exists(self.dst_voc_dir):
            if self.verbose:
                print "Creating directory: " + "\033[1m" + self.dst_voc_dir + "\033[0m"
            os.makedirs(self.dst_voc_dir)
        if not os.path.exists(self.dst_coo_dir):
            if self.verbose:
                print "Creating directory: " + "\033[1m" + self.dst_coo_dir + "\033[0m"
            os.makedirs(self.dst_coo_dir)


    def run(self):

        # automatically calculate window size if auto_window is not None
        if self.auto_window:
            self.get_reviews()
            self.get_sentiment_words()
            self.get_avg_nearest_sentiment_distance()

        if self.verbose:
            print "\nConstructing co-occurrence network with window size " + str(self.window_size)

        self.create_dirs()

        if self.verbose:
            print "Fetching vocab.."
        vocab = self.get_or_build(self.vocab_path, self.build_vocab, self.corpus)

        if self.verbose:
            print "Vocab has " + str(len(vocab)) + " elements."
            print "Saving vacabularies to: " + self.dst_voc_dir + "\033[1m" + self.filename + "\033[0m"

        inv_vocab={}
        string = ""
        with open(os.path.join(self.dst_voc_dir, self.filename), "w") as fp:
            for key, value in vocab.items():
                fp.write("%s %s\n" % (value[0], key))
                inv_vocab[value[0]]=key


        if self.verbose:
            print "Fetching cooccurrence list.."
        self.corpus.seek(0)
        cooccurrences = self.get_or_build(self.cooccur_path,
                                     self.build_cooccur, vocab, self.corpus,
                                     window_size=self.window_size,
                                     min_count=self.min_count)

        if self.verbose:
            print "Saving co-occurrence network to: " + self.dst_coo_dir + "\033[1m" + self.filename + "\033[0m"
        with open(self.dst_coo_path, "w") as fp:
            fp.write("\n".join("%s %s %s" % (inv_vocab[x[0]], inv_vocab[x[1]], x[2]) for x in cooccurrences))


    def get_reviews(self):
        """ load reviews from `data/reviews/` """

        if self.verbose:
            print "Loading reviews from " + "\033[1m" + self.src + "\033[0m"
        with open(self.src) as f:
            self.reviews = f.readlines()


    def get_sentiment_words(self):
        """ load sentiment_words from data/lexicon/candidates.json """

        if self.verbose:
            print "Loading sentiment words from " + "\033[1m" + self.src_candidates + "\033[0m"
        with open(self.src_candidates) as f:
            self.sentiment_words = json.load(f)


    def get_avg_nearest_sentiment_distance(self):
        """ walk into reviews and get distance between entity and sentiment_word """

        if self.verbose:
            print "Using sentiment words to calculate average nearest sentiment distance"

        sentiment_stemmed_words = []
        for sentiment_words in self.sentiment_words:
            for word in sentiment_words:
                sentiment_stemmed_words.append(word["stemmed_word"])

        review_cnt = 0
        review_length = len(self.reviews)
        sentiment_distance_list = []
        for review in self.reviews:
            review_cnt += 1
            words = review.split(" ")[1:-1]
            num_of_words = len(words)

            indexes = [i for i,w in enumerate(words) if "STAR_" in w]
            for index in indexes:
                nearest_sentiment_distance = 1
                while index+nearest_sentiment_distance < num_of_words or\
                      index-nearest_sentiment_distance >= 0:
                        # forward search
                        if index+nearest_sentiment_distance < num_of_words:
                            if words[index+nearest_sentiment_distance] in sentiment_stemmed_words:
                                break
                        # backward search
                        if index-nearest_sentiment_distance >= 0:
                            if words[index-nearest_sentiment_distance] in sentiment_stemmed_words:
                                break
                        nearest_sentiment_distance += 1

                sentiment_distance_list.append(nearest_sentiment_distance)

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (review_cnt, review_length))
                sys.stdout.flush()

        self.nearest_sentiment_distance = float(sum(sentiment_distance_list) / float(len(sentiment_distance_list)))
        if self.nearest_sentiment_distance > int(self.nearest_sentiment_distance):
            self.window_size = int(self.nearest_sentiment_distance) + 1
        else:
            self.window_size = int(self.nearest_sentiment_distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="../data/", help="data source folder")
    parser.add_argument("--min_count", type=int, default=5, help="words with the count less than the minimum count are excluded")
    parser.add_argument("--window_size", type=int, default=2, help="window size for constructing networks")
    parser.add_argument("--vocab_path", type=str, default=None, help="path for vocabularies in the network")
    parser.add_argument("--cooccur_path", type=str, default=None, help="path for cooccurrence networks")
    parser.add_argument("--auto_window", default=False, help="automatically calculate the window size or not", action="store_true")
    parser.add_argument("--verbose", default=False, help="verbose logging or not", action="store_true")
    args = parser.parse_args()

    main = ConstructNetwork(src_folder=args.src_folder, min_count=args.min_count, window_size=args.window_size,\
            vocab_path=args.vocab_path, cooccur_path=args.cooccur_path, auto_window=args.auto_window, verbose=args.verbose)
    main.run()
