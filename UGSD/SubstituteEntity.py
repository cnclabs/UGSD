# -*- coding: utf-8 -*-
import argparse
import sys, re, json, os, uuid, itertools
from operator import itemgetter
from collections import OrderedDict
import unicodedata, linecache
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import corenlp

class SubstituteEntity:
    """ This program aims to recognize entities in `src_folder/reviews/`
        and replace them with corresponding rating symbols.
        The results will be stored in `src_folder/starred_reviews/`
    """

    def __init__(self, src, src_folder="../data/", corenlp_path="../stanford-corenlp/",\
                 regexp=None, verbose=False):

        # initialization
        self.src = src
        self.filename = re.search("([A-Za-z|.]+\-*[A-Za-z|.]+\-*[A-Za-z|.]+\_.*).json", self.src).group(1)
        self.src_candidates = os.path.join(src_folder, "lexicon/candidates.json")
        self.dst_starred = os.path.join(src_folder, "starred_reviews/")
        self.corenlp_path = os.path.normpath(corenlp_path)
        self.regexp = regexp
        self.verbose = verbose

        self.entity = {}
        self.entity_name = ""
        self.entity_regexp = []
        self.ratings = []
        self.sentiment_words = []
        self.adv_adj_combinations = {}
        self.clean_reviews = []
        self.starred_reviews = []

        self.stopwords = set(stopwords.words("english"))
        self.stopwords.remove("not")
        self.stemmer = SnowballStemmer("english")

        # need set the CORENLP_HOME path
        os.environ["CORENLP_HOME"] = self.corenlp_path
        self.corenlp = corenlp.CoreNLPClient(annotators="tokenize ssplit dcoref".split(), timeout=50000)
        ### self.corenlp = corenlp.CoreNLPClient(annotators="tokenize ssplit dcoref".split(), timeout=50000, endpoint="http://localhost:8080")


    def get_entity(self):
        """ Load data from data/reviews/*.json """

        if self.verbose:
            print "Loading data from " + "\033[1m" + self.src + "\033[0m"
        with open(self.src) as f:
            self.entity = json.load(f)


    def get_entity_name(self):
        """ get entity_name """

        self.entity_name = self.entity["entity"].replace("-"," ").replace("&","and").replace(" "+self.entity["category"],"")
        if self.verbose:
            print "This is entity: " + "\033[1m" + self.entity_name + "\033[0m"


    def get_entity_regexp(self):
        """ entity_regexp is the regular expression for different entities """

        if self.verbose:
            print "\n" + "-"*80
            print "Generating entity regular expression"
        if self.regexp != None:
            self.entity_regexp = self.regexp(self.entity["category"], self.entity_name)
            return


    def get_sentiment_words(self):
        """ load sentiment words from data/lexicon/candidates.json """

        if self.verbose:
            print "\n" + "-"*80 + "\n" + "Loading sentiment words from " + "\033[1m" + self.src_candidates + "\033[0m"
        with open(self.src_candidates) as f:
            self.sentiment_words = json.load(f)


    def get_clean_reviews(self):
        """ preprocess reviews """

        if self.verbose:
            print "\n" + "-"*80 + "\nCleaning reviews"

        reviews = self.entity["reviews"]
        cnt = 0
        review_length = len(reviews)
        for review_dict in reviews:
            cnt += 1
            text = review_dict["review"]

            # remove accents
            text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore")
            # remove all website urls written in the review
            text = re.sub(r"https?:\/\/.*[\r\n]*", " ", text, flags=re.MULTILINE)
            # remove non english letters or words and numbers
            text = re.sub(r"[^a-zA-Z!@#$%^&*():;/\\<>\"+_\-.,?=\s\|\']", "", text)
            # remove extra nextline
            text = re.sub(r"(\\n)+", r" ", text)

            # I'm -> I am
            text = re.sub(r"'m ", " am ", text)
            text = re.sub(r"'re ", " are ", text)
            text = re.sub(r"'s ", " is ", text)
            text = re.sub(r"'ve ", " have ", text)
            text = re.sub(r"'d ", " would ", text)
            text = re.sub(r" won't ", " will not ", text)
            text = re.sub(r"n't ", " not ", text)
            text = re.sub(r"'ll ", " will ", text)

            # space out every sign & symbol & punctuation
            text = re.sub(r"([^\w\s]|\_)",r" \1 ", text)
            text = text.replace("\'","")
            # remove ` - `, ` -`, `- `
            text = re.sub(r"(\-)+", "-", text)
            text = re.sub(r"(\s)+\-(\s)+|(\s)+\-|\-(\s)+|(\A)\-|\-(\Z)", " ", text)
            # turn multiple spaces into one
            text = re.sub(r"(\s)+", r" ", text)

            self.clean_reviews.append(text)
            self.ratings.append(review_dict["rating"])

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (cnt, review_length))
                sys.stdout.flush()


    def get_combinations(self):
        """ load sentiment words which are parts of adv-adj combinations """

        if self.verbose:
            print "\n" + "-"*80 + "\n" + "Loading adv-adj combinations"

        for sentiment_words in self.sentiment_words:
            for word in sentiment_words:
                if "_" in word["stemmed_word"]:
                    for w in word["word"]:
                        wl = w.split("_")
                        for i in xrange(1, len(wl)):
                            front_word = "_".join(wl[:i])
                            if front_word not in self.adv_adj_combinations:
                                self.adv_adj_combinations[front_word] = []
                            self.adv_adj_combinations[front_word].append(wl[i])


    def dcoref_entity_linking(self, text, entity_star):
        try:
            annotate = self.corenlp.annotate(text)
        except:
            return text.lower()
        chains = annotate.corefChain
        sentences = annotate.sentence
        tokens = []

        # get coreference list
        dcoref = []
        for i in xrange(len(chains)):
            if len(chains[i].mention) > 1:
                dcoref.append({})
                for mention in chains[i].mention:
                    if (mention.sentenceIndex, mention.beginIndex) in dcoref[-1]:
                        if mention.endIndex < dcoref[-1][(mention.sentenceIndex, mention.beginIndex)]:
                            dcoref[-1][(mention.sentenceIndex, mention.beginIndex)] = mention.endIndex
                    else:
                        dcoref[-1][(mention.sentenceIndex, mention.beginIndex)] = mention.endIndex

        # get the list which refers to the entity
        dcoref_entity = []
        entity_loc = []
        if len(dcoref) > 0:
            for i in sentences:
                tokens.append([])
                for j in i.token:
                    tokens[-1].append(j.word.lower())

            for reg in self.entity_regexp:
                for d in dcoref:
                    for (sen, beg), end in d.iteritems():
                        ref_text = " " + " ".join(tokens[sen][beg:end]) + " "
                        # if match regexp, copy dcoref list to dcoref_entity
                        if bool(re.search(reg, ref_text)):
                            dcoref_entity.append([])
                            entity_loc.append([])
                            for (s, b), e in d.iteritems():
                                ref_text = " " + " ".join(tokens[s][b:e]) + " "
                                if bool(re.search(reg, ref_text)):
                                    entity_loc[-1].append((s, b, e))
                                else:
                                    dcoref_entity[-1].append((s, b, e))
                            if len(dcoref_entity[-1]) == 0:
                                del dcoref_entity[-1]
                            else:
                                dcoref_entity[-1] = sorted(dcoref_entity[-1], key=lambda tup: (tup[0], tup[1], tup[2]), reverse=True)
                            break

        # delete those not referring to the entity
        del_entity_list = []
        if len(dcoref_entity) > 1:
            for i in xrange(len(dcoref_entity)):
                for j in dcoref_entity[i]:
                    for k in xrange(len(dcoref_entity)):
                        if i == k:
                            continue
                        for m in dcoref_entity[k]:
                            if j[0] == m[0] and\
                               (j[1] >= m[1] and j[2] <= m[2]):
                                if k not in del_entity_list:
                                    del_entity_list.append(k)
                                break
        if len(del_entity_list) != 0:
            del_entity_list.sort(reverse=True)
            for i in del_entity_list:
                del dcoref_entity[i]

        # replace the dcoref list, the same as entity substitution
        # and update the text
        if len(dcoref_entity) != 0:
            dcoref_entity = [item for l in dcoref_entity for item in l]
            dcoref_entity = sorted(dcoref_entity, key=lambda tup: (tup[0], tup[1], tup[2]), reverse=True)

            for (s, b, g) in dcoref_entity:
                tokens[s][b] = entity_star
                del tokens[s][b+1:g]

            # generate the new review text
            # after substitute those referring to the entity
            new_text_list = []
            for t in tokens:
                new_text_list.append(" ".join(t))
            text = " ".join(new_text_list)
        else:
            text = text.lower()

        return text


    def get_starred_reviews(self):
        """ match the entity_name in the reviews with entity_regexp and replace them by entity_al  """

        if self.verbose:
            print "\n" + "-"*80 + "\n" + "Processing backend reviews"

        review_cnt = 0
        review_length = len(self.clean_reviews)
        for review, rating in zip(self.clean_reviews, self.ratings):
            review_cnt += 1

            # remove all punctuations except for , . ? ! ; : and -
            # because these are important factors for coreference resolution
            # -: composite adj.
            review = re.sub(r"[^\w\s,.?!;:\-]|\_", r" ", review)
            # remove extra spaces
            review = re.sub(r"(\s)+", r" ", review)
            # substitue those referring to the entity
            # by coreference resolution
            review = self.dcoref_entity_linking(review, "STAR_"+str(rating))

            # remove all punctuations
            review = re.sub(r"[^\w\s\-\_]", r" ", review)
            # remove extra spaces
            review = re.sub(r"(\s)+", r" ", review)

            # entity substitution
            for reg in self.entity_regexp:
                review = re.sub(reg, " STAR_"+str(rating)+" ", review)

            # remove extra spaces
            review = re.sub(r"(\s)+", r" ", review)
            # split review into a list of words
            words = review.split(" ")
            # remove stopwords
            words_stopwords_removed = [w for w in words if w not in self.stopwords]

            words_comb = []
            i = 0
            while i < len(words_stopwords_removed):
                if i + 1 < len(words_stopwords_removed) and\
                   words_stopwords_removed[i] in self.adv_adj_combinations and\
                   words_stopwords_removed[i+1] in self.adv_adj_combinations[words_stopwords_removed[i]]:
                    front = words_stopwords_removed[i] + "_" + words_stopwords_removed[i+1]
                    j = i + 2
                    while j < len(words_stopwords_removed) and\
                          front in self.adv_adj_combinations and\
                          words_stopwords_removed[j] in self.adv_adj_combinations[front]:
                        front += "_" + words_stopwords_removed[j]
                        j += 1
                    words_comb.append(front)
                    i = j
                else:
                    words_comb.append(words_stopwords_removed[i])
                    i += 1

            words_stemmed = [self.stemmer.stem(w) if "STAR_"+str(rating) != w else w for w in words_comb]
            review = " " + " ".join(words_stemmed).encode("utf-8").strip() + " "
            self.starred_reviews.append(review)

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (review_cnt, review_length))
                sys.stdout.flush()

        # stop stanford corenlp server
        self.corenlp.stop()


    def create_dirs(self, category):
        """ create directory under data/backend_revies/ """

        if self.verbose:
            print "Creating directories if not existing"
        dir1 = os.path.dirname(os.path.join(self.dst_starred, category+"/"))
        if not os.path.exists(dir1):
            if self.verbose:
                print "Creating Directory: " + dir1
            os.makedirs(dir1)


    def render(self):
        """ render processed reviews """

        reload(sys)
        sys.setdefaultencoding("utf-8")

        if self.verbose:
            print "\n" + "-"*80 + "\n" + "Saving json files"

        self.create_dirs(self.entity["category"])
        result_file = open(os.path.join(self.dst_starred, self.entity["category"], self.filename+".txt"), "w+")
        for review in self.starred_reviews:
            result_file.write(review.encode("utf-8") + "\n")
        result_file.close()

        if self.verbose:
            print self.filename, "is complete"


    def run(self):
        print "Processing " +"\033[1m" + self.filename + ".json"  + "\033[0m"

        self.get_entity()
        self.get_entity_name()
        self.get_entity_regexp()
        self.get_clean_reviews()
        self.get_sentiment_words()
        self.get_combinations()
        self.get_starred_reviews()
        self.render()


    def PrintException(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print '    Exception in ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


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


def TripAdvisor_regexp(category, entity_name):
    """ custom function for regular expression generation for reviews in different reviews  """
    entity_regexp_list = []
    # TripAdvisor
    entity_regexp = entity_name.lower()
    entity_regexp = re.sub(r"[^a-zA-Z!@#$%^&*():;/\\<>\"+_\-.,?=\s\|]", "", entity_regexp)
    entity_regexp = entity_regexp.split()
    if entity_regexp[-1] == "tours":
        entity_regexp[0] = "\\s(the\\s|this\\s|" + entity_regexp[0]

        for i in xrange(len(entity_regexp)-1):
            entity_regexp[i] += "\\s"
        for i in xrange(len(entity_regexp)-2):
            entity_regexp[i] += "|"
        entity_regexp[len(entity_regexp)-2] = entity_regexp[len(entity_regexp)-2] + ")*"

        entity_regexp[-1] = entity_regexp[-1][:-1] # tours -> tour
        entity_regexp = "".join(entity_regexp)
        entity_regexp_list.append(entity_regexp + "(s)?\\s")
    else:
        if len(entity_regexp) > 1:
            entity_regexp[0] = "\\s(the\\s|this\\s|" + entity_regexp[0]

            for i in xrange(len(entity_regexp)-1):
                if entity_regexp[i][-1] == "s":
                    entity_regexp[i] += "(s)?\\s"
                else:
                    entity_regexp[i] += "\\s"
            for i in xrange(len(entity_regexp)-2):
                entity_regexp[i] += "|"
            entity_regexp[len(entity_regexp)-2] = entity_regexp[len(entity_regexp)-2] + ")+"

        if entity_regexp[-1][-1] == "s":
            entity_regexp[-1] = entity_regexp[-1][:-1] + "(s)?"# temples -> temple(s)
        entity_regexp[-1] = "(" + entity_regexp[-1] + "|place)"
        entity_regexp_list.append("".join(entity_regexp))

    return entity_regexp_list


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="path for the review")
    parser.add_argument("--src_folder", type=str, default="../data/", help="data source folder")
    parser.add_argument("--corenlp_path", type=str, default="../stanford-corenlp/", help="the path for Stanford CoreNLP")
    parser.add_argument("--verbose", default=False, help="verbose logging or not", action="store_true")
    args = parser.parse_args()

    main = SubstituteEntity(src=args.src, src_folder=args.src_folder, corenlp_path=args.corenlp_path,\
                            regexp=TripAdvisor_regexp, verbose=args.verbose)
    main.run()

