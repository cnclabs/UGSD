# -*- coding: utf-8 -*-
import argparse
import json, os, sys, uuid, re
from collections import OrderedDict
import numpy as np
from operator import itemgetter

class MergeCorpus:
    """ This program aims to
        merge `src_folder/starred_reviews/category/*.txt` into `src_folder/corpus/starred_reivews.txt`
    """

    def __init__(self, src_folder="../data/", verbose=False):

        # initialization
        self.src_sr = os.path.join(src_folder, "starred_reviews/")
        self.dst_c = os.path.join(src_folder, "corpus/starred_reviews.txt")
        self.verbose = verbose


    def merge_starred_reviews(self):
        """ load and merge all reviews in data/starred_reviews/ into one single file named 'corpus/starred_reviews.txt' """

        if self.verbose:
            print "Merging starred reviews"

        starred_reviews = []
        for dirpath, dir_list, file_list in os.walk(self.src_sr):
            if self.verbose:
                print "Walking into directory: " + str(dirpath)

            if len(file_list) > 0:
                if self.verbose:
                    print "Files found: " + "\033[1m" + str(file_list) + "\033[0m"
                file_cnt = 0
                length = len(file_list)
                for f in file_list:
                    # in case there is a goddamn .DS_Store file
                    if str(f) == ".DS_Store":
                        if self.verbose:
                            print "Removing " + str(os.path.join(dirpath, f))
                        os.remove(os.path.join(dirpath, f))
                    else:
                        file_cnt += 1
                        with open(os.path.join(dirpath, f)) as fp:
                            starred_reviews.append(fp.read())
            else:
                if self.verbose:
                    print "No file is found"

        self.render_starred_corpus(starred_reviews)


    def render_starred_corpus(self, starred_reviews):
        """ all starred_backend_reviews in all category_*.txt -> data/corpus/starred_reviews.txt """

        if self.verbose:
            print "Saving data to: " + "\033[1m" + self.dst_c + "\033[0m"
        review_cnt = 0
        corpus_length = len(starred_reviews)
        f_starred_reviews = open(self.dst_c, "w+")
        for review in starred_reviews:
            review_cnt += 1
            f_starred_reviews.write(review)

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (review_cnt, corpus_length))
                sys.stdout.flush()
        f_starred_reviews.close()


    def create_dirs(self):
        """ create the directory if not exist"""

        dir1 = os.path.dirname(self.dst_c)
        if self.verbose:
            print "Creating directories if not existing"
        if not os.path.exists(dir1):
            if self.verbose:
                print "Creating directory: " + dir1
            os.makedirs(dir1)


    def run(self):
        self.create_dirs()
        self.merge_starred_reviews()


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
    parser.add_argument("--verbose", default=False, help="verbose logging or not", action="store_true")
    args = parser.parse_args()

    main = MergeCorpus(src_folder=args.src_folder, verbose=args.verbose)
    main.run()
