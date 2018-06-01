# -*- coding: utf-8 -*-
import argparse
import sys, os, json, uuid, re
from collections import OrderedDict
import numpy as np
from scipy import spatial
from operator import itemgetter

class ConstructLexicon:
    """ This program aims to calculate consine similarity
        and pick up the top N sentiment words for different ratings
    """

    def __init__(self, src_folder="../data/",\
                 min_star=1, max_star=5, star_scale=1,\
                 std_threshold=1.0, verbose=False):

        # initialization
        self.src = os.path.join(src_folder, "network/cooccur/starred_repr.txt")
        self.src_candidates = os.path.join(src_folder, "lexicon/candidates.json")
        self.dst = os.path.join(src_folder, "starred_lexicons/")
        self.dst_max = os.path.join(self.dst, "max-scheme/")
        self.dst_zscore = os.path.join(self.dst, "zscore-scheme/")

        self.min_star = min_star
        self.max_star = max_star
        self.star_scale = star_scale
        self.std_threshold = std_threshold
        self.verbose = verbose

	self.unique_words = {}
        self.representations = []
        self.sentiment_words = []
        self.matrix = {}
        self.max_scheme_lexicons = [[] for _ in xrange(max_star - min_star + 1)]
        self.mean = []
        self.std = []
        self.zscore_scheme_lexicons = [[] for _ in xrange(max_star - min_star + 1)]


    def get_representations(self):
	""" first call readline() to read the first line of representations file to get vocab_size and dimension_size """

        if self.verbose:
            print "Loading representations of words from " + "\033[1m" + self.src + "\033[0m"

        f_src = open(self.src, "r")
        vocab_size, dimension_size = f_src.readline().split(" ")
        for index in range(int(vocab_size)):
            line = f_src.readline().split(" ")
            self.unique_words[line[0]] = index
            self.representations.append([float(i) for i in line[1:]])

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (index+1, int(vocab_size)))
                sys.stdout.flush()
        f_src.close()


    def get_candidate_words(self):
	""" get candidate words """

        if self.verbose:
            print "\nLoading data from " + "\033[1m" + self.src_candidates + "\033[0m"

        with open(self.src_candidates, "r") as f:
            lexicon = json.load(f)

        if self.verbose:
            print "Building sentiment words and representations"

        index = 0
        length = len(lexicon[0])
        for word in lexicon[0]:
            index += 1
            if word["stemmed_word"] in self.unique_words:
                self.sentiment_words.append(word)

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (index, length))
                sys.stdout.flush()


    def get_distance_matrix(self):
        """ accumulate the distance matrix """

        senti_len = len(self.sentiment_words)
        for i in xrange(len(self.sentiment_words)):
            self.matrix[self.sentiment_words[i]["stemmed_word"]] = []
            for star_now in range(self.min_star, self.max_star+1, self.star_scale):
                self.matrix[self.sentiment_words[i]["stemmed_word"]].append(1-spatial.distance.cosine(self.representations[self.unique_words["STAR_"+str(star_now)]], self.representations[self.unique_words[self.sentiment_words[i]["stemmed_word"]]]))

                if self.verbose:
                    sys.stdout.write("\rStatus: %s / %s @ %s / %s" % (star_now, self.max_star, i+1, senti_len))
                    sys.stdout.flush()


    def get_max_scheme_lexicons(self):
        """ get the lexicons by maximum-cosine-similarity scheme """

        if self.verbose:
            print "\nGeting the lexicons by maximum-cosine-similarity scheme"

        senti_len = len(self.sentiment_words)
        for i in xrange(len(self.sentiment_words)):
            index, value = max(enumerate(self.matrix[self.sentiment_words[i]["stemmed_word"]]), key=itemgetter(1))
            self.max_scheme_lexicons[index].append((self.sentiment_words[i], value))

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (i+1, senti_len))
                sys.stdout.flush()

        for i in range(len(self.max_scheme_lexicons)):
            self.max_scheme_lexicons[i] = sorted(self.max_scheme_lexicons[i], key=itemgetter(1), reverse=True)

        self.render(self.max_scheme_lexicons, self.dst_max)


    def calculate_zscore(self):
        """ calculate z-score for all columns """

        if self.verbose:
            print "\nCalculating z-score for all columns"

        senti_len = len(self.sentiment_words)
        all_star = self.max_star - self.min_star + 1
        for star_now in xrange(all_star):
            vec = []
            for i in xrange(senti_len):
                vec.append(self.matrix[self.sentiment_words[i]["stemmed_word"]][star_now])
            self.mean.append(np.mean(np.array(vec)))
            self.std.append(np.std(np.array(vec)))

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (star_now+1, all_star))
                sys.stdout.flush()


    def get_zscore_scheme_lexicons(self):
        """ get the lexicons by z-score scheme """

        if self.verbose:
            print "\nGeting the lexicons by z-score scheme"

        self.calculate_zscore()

        senti_len = len(self.sentiment_words)
        all_star = self.max_star - self.min_star + 1
        for i in xrange(senti_len):
            for star_now in xrange(all_star):
                if self.matrix[self.sentiment_words[i]["stemmed_word"]][1] > self.mean[star_now] + self.std_threshold * self.std[star_now]:
                    self.zscore_scheme_lexicons[star_now].append((self.sentiment_words[i], self.matrix[self.sentiment_words[i]["stemmed_word"]][star_now]))

                if self.verbose:
                    sys.stdout.write("\rStatus: %s / %s @ %s / %s" % (star_now+1, all_star, i+1, senti_len))
                    sys.stdout.flush()

        for i in range(len(self.zscore_scheme_lexicons)):
            self.zscore_scheme_lexicons[i] = sorted(self.zscore_scheme_lexicons[i], key=itemgetter(1), reverse=True)

        self.render(self.zscore_scheme_lexicons, self.dst_zscore)


    def render(self, sentiment_words, dst_path):
        """ save every sentiment words for lexicons of different ratings """

        if self.verbose:
            print "\nConstructing lexicons"

        self.create_dirs(self.dst)
        self.create_dirs(dst_path)

        for star_now in range(self.min_star, self.max_star+1, self.star_scale):
            star_lexicon = OrderedDict()
            star_lexicon["rating"] = str(star_now)
            ordered_sentiment_words = []
            index = 0
            for word_dict in sentiment_words[star_now - self.min_star]:
                index += 1
                word = OrderedDict()
                word["index"] = index
                word["cos_sim"] = word_dict[1]
                word["count"] = word_dict[0]["count"]
                word["stemmed_word"] = word_dict[0]["stemmed_word"]
                word["word"] = word_dict[0]["word"]
                ordered_sentiment_words.append(NoIndent(word))

            star_lexicon["sentiment_words"] = ordered_sentiment_words

            if self.verbose:
                print "Saving lexicon to " + str(os.path.join(dst_path, "star_"+str(star_now)+".json"))

            with open(os.path.join(dst_path, "star_"+str(star_now)+".json"), "w") as f:
                f.write(json.dumps(star_lexicon, indent=4, cls=NoIndentEncoder))


    def create_dirs(self, dst):
        """ create the directory if not exist """

        dir1 = os.path.dirname(dst)
        if not os.path.exists(dir1):
            if self.verbose:
                print "\nCreating directory: " + dir1
            os.makedirs(dir1)


    def run(self):

        self.get_representations()
        self.get_candidate_words()
        self.get_distance_matrix()
        self.get_max_scheme_lexicons()
        self.get_zscore_scheme_lexicons()


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}


    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)


    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in self._replacement_map.iteritems():
            result = result.replace('"@@%s@@"' % (k,), v)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="../data/", help="data source folder")
    parser.add_argument("--min_star", type=int, default=1, help="minimum rating scale")
    parser.add_argument("--max_star", type=int, default=5, help="maximum rating scale")
    parser.add_argument("--star_scale", type=int, default=1, help="rating scale")
    parser.add_argument("--std_threshold", type=float, default=1.0, help="the threshold for the z-score scheme")
    parser.add_argument("--verbose", default=False, help="verbose logging or not", action="store_true")
    args = parser.parse_args()

    main = ConstructLexicon(src_folder=args.src_folder, min_star=args.min_star, max_star=args.max_star,\
                            star_scale=args.star_scale, std_threshold=args.std_threshold, verbose=args.verbose)
    main.run()
