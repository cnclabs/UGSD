# -*- coding: utf-8 -*-
import argparse
import json, sys, uuid, re, os, subprocess, csv, time, signal
from collections import OrderedDict
import unicodedata, linecache
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag.stanford import CoreNLPPOSTagger, StanfordNERTagger

class SelectCandidates:
    """ This program aims to select candidate words from reviews
        We picks up sentiment words and handles the negation problem
        The result will be stored in `src_folder/lexicon/candidates.json`
    """

    def __init__(self, src_folder="../data/", freq_thre=100, corenlp_path="../stanford-corenlp/",\
                 ner_path="../stanford-ner/", verbose=False):

        # initialization
        self.src = os.path.join(src_folder, "reviews/")
        self.corenlp_path = os.path.normpath(corenlp_path) + "/"
        self.stanford_ner_path = os.path.normpath(ner_path) + "/"
        self.frequency_threshold = freq_thre
        self.dst = os.path.join(src_folder, "lexicon/candidates.json")
        self.dst_allReviews  = os.path.join(src_folder, "allReviews/")
        self.dst_ner_tsv = os.path.join(src_folder, "ner_tsv/")
        self.dst_ne = os.path.join(src_folder, "ne/")
        self.verbose = verbose

        # pick up sentiment words
        self.pos_tags = ["JJ","JJR", "JJS", "RB", "RBR", "RBS"]
        self.pos_tagged_statistics = {}

        # it is based on CoreNLP, a new version of stanford pos tagger
        self.pos_tagger = CoreNLPPOSTagger()
        self.stemmer = SnowballStemmer("english")
        self.stopwords = set(stopwords.words("english"))
        # remove `not` because we need combine `not` and sentiment words
        self.stopwords.remove("not")


    def stanford_ner(self):
        """ call stanford java ner api """

        self.merge_reviews()
        self.run_ner()
        self.find_named_entity()


    def merge_reviews(self):
        """ merge all reviews for named entity recognition """

        if self.verbose:
            print "Merging all reviews for named entity recognition" + "\n" + "-"*80

        self.create_dir(self.dst_allReviews)

        for dirpath, dirs, files in os.walk(self.src):
            for f in files:
                filename = re.search("([A-Za-z|.]+\-*[A-Za-z|.]+\-*[A-Za-z|.]+\_.*).json", f).group(1)
                data = json.load(open(os.path.join(dirpath, f)))
                with open(os.path.join(self.dst_allReviews, filename+".txt"), "w+") as rf:
                    for r in data["reviews"]:

                        text = r["review"]
                        # remove accents
                        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore")
                        # remove all website urls written in the review
                        text = re.sub(r"https?:\/\/.*[\r\n]*", " ", text, flags=re.MULTILINE)
                        # remove non english letters or words and numbers
                        text = re.sub(r"[^a-zA-Z!@#$%^&*():;/\\<>\"+_\-.,?=\s\|\']", "", text)
                        # remove extra nextline
                        text = re.sub("(\\n)+", r" ", text)

                        # I'm -> I am
                        text = re.sub(r"'m ", " am ", text)
                        text = re.sub(r"'re ", " are ", text)
                        text = re.sub(r"'s ", " is ", text)
                        text = re.sub(r"'ve ", " have ", text)
                        text = re.sub(r"'d ", " would ", text)
                        text = re.sub(r" won't ", " will not ", text)
                        text = re.sub(r"n't ", " not ", text)
                        text = re.sub(r"'ll ", " will ", text)

                        # remove all punctuations except for , . ? ! ; : and -
                        # -: composite adj.
                        text = re.sub("[^\w\s,.?!;:\-]|\_", r" ", text)

                        # Space out every sign & symbol & punctuation
                        text = re.sub("([^\w\s])", r" \1 ", text)

                        text = text.replace("\'", "")
                        # remove ` - `, ` -`, `- `
                        text = re.sub(r"(\-)+", "-", text)
                        text = re.sub(r"(\s)+\-(\s)+|(\s)+\-|\-(\s)+|(\A)\-|\-(\Z)", " ", text)
                        # turn multiple spaces into one
                        text = re.sub(r"(\s)+", " ", text)
                        # remove extra space at both ends of the text
                        text = text.strip()

                        rf.write(text)
                        rf.write("\n\n. CHANGE-REVIEW .\n\n")


    def run_ner(self):
        """ run shell to call NER """

        if self.verbose:
            print "Running shell to call Stanford NER" + "\n" + "-"*80

        self.create_dir(self.dst_ner_tsv)

        comm = "java -mx1g -cp \"%s*:%slib/*\" edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier %sclassifiers/english.all.3class.distsim.crf.ser.gz -outputFormat tabbedEntities -textFile %s > %s"
        for dirpath, dirs, files in os.walk(self.dst_allReviews):
            for f in files:
                filename = re.search("([A-Za-z|.]+\-*[A-Za-z|.]+\-*[A-Za-z|.]+\_.*).txt", f).group(1)
                src_file = os.path.join(dirpath, f)
                dst_file = os.path.join(self.dst_ner_tsv, filename+".tsv")
                command = comm % (self.stanford_ner_path, self.stanford_ner_path, self.stanford_ner_path, src_file, dst_file)
                subprocess.call(command, shell=True)


    def find_named_entity(self):
        """ find named entity from the ner tsv """

        if self.verbose:
            print "Finding named entity from ner tsv files" + "\n" + "-"*80

        self.create_dir(self.dst_ne)

        for dirpath, dirs, files in os.walk(self.dst_ner_tsv):
            for f in files:
                filename = re.search("([A-Za-z|.]+\-*[A-Za-z|.]+\-*[A-Za-z|.]+\_.*).tsv", f).group(1)
                src_file = os.path.join(dirpath, f)
                dst_file = os.path.join(self.dst_ne, filename+".txt")
                rs = [set()]

                with open(src_file, "rb") as tsvin:
                    data = csv.reader(tsvin, delimiter="\t")
                    for r in data:
                        if len(r) != 0 and r[0] != "":
                            if r[1] == "ORGANIZATION" or r[1] == "PERSON" or r[1] == "LOCATION":
                                l = r[0].split(" ")
                                for i in l:
                                    if (i, r[1]) not in rs:
                                        rs[-1].add((i, r[1]))
                        elif len(r) > 2 and "CHANGE-REVIEW" in r[2]:
                            rs.append(set())

                with open(dst_file, "w+") as rf:
                    for rs_index in range(len(rs)-1):
                        rf.write(str(rs_index)+",FILEINDEX\n")
                        for i in rs[rs_index]:
                            rf.write(i[0]+","+i[1]+"\n")


    def get_sentiment_words(self):
        """ load all reviews in src folder: data/reviews/ and merge them """

        # start Stanford CoreNLP server in a new process
        comm = "java -mx4g -cp \"%s*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 50000"
        command = comm % (self.corenlp_path)
        proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        time.sleep(10) # wait for starting Stanford CoreNLP server

        for dirpath, dir_list, file_list in os.walk(self.src):
            if self.verbose:
                print "Walking into directory: " + str(dirpath)

            if len(file_list) > 0:
                for f in file_list:
                    # in case there is a goddamn .DS_Store file
                    if str(f) == ".DS_Store":
                        if self.verbose:
                            print "Removing " + dirpath + "/" + str(f)
                        os.remove(os.path.join(dirpath, f))
                    else:
                        with open(os.path.join(dirpath, f)) as fp:
                            entity = json.load(fp)

                    if self.verbose:
                        print "Processing " + "\033[1m" + entity["entity"] + "\033[0m" + " in " + "\033[1m" + entity["category"] + "\033[0m"
                    self.analyze_part_of_speech(entity["reviews"], f)
            else:
                if self.verbose:
                    print "No file is found in " + str(dirpath)

        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        if self.verbose:
            print "Part of Speech Analysis on Reviews are Done"
            print "-"*80


    def analyze_part_of_speech(self, reviews, filename):
        """ run nltk.pos_tag to analysis the part_of_speech of every word """

        ner_set = self.load_ner_tags(filename)

        for review_index in range(len(reviews)):

            text = reviews[review_index]["review"]
            # remove accents
            text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore")
            # remove all website urls written in the review
            text = re.sub(r"https?:\/\/.*[\r\n]*", " ", text, flags=re.MULTILINE)
            # remove non english letters or words and numbers
            text = re.sub(r"[^a-zA-Z!@#$%^&*():;/\\<>\"+_\-.,?=\s\|\']", "", text)
            # remove extra nextline
            text = re.sub("(\\n)+", r" ", text)

            # I'm -> I am
            text = re.sub(r"'m ", " am ", text)
            text = re.sub(r"'re ", " are ", text)
            text = re.sub(r"'s ", " is ", text)
            text = re.sub(r"'ve ", " have ", text)
            text = re.sub(r"'d ", " would ", text)
            text = re.sub(r" won't ", " will not ", text)
            text = re.sub(r"n't ", " not ", text)
            text = re.sub(r"'ll ", " will ", text)

            # remove all punctuations except for , . ? ! ; : and -
            # -: composite adj.
            text = re.sub("[^\w\s,.?!;:\-]|\_", r" ", text)

            # space out every sign & symbol & punctuation
            text = re.sub("([^\w\s])", r" \1 ", text)

            text = text.replace("\'", "")
            # remove ` - `, ` -`, `- `
            text = re.sub(r"(\-)+", "-", text)
            text = re.sub(r"(\s)+\-(\s)+|(\s)+\-|\-(\s)+|(\A)\-|\-(\Z)", " ", text)
            # turn multiple spaces into one
            text = re.sub(r"(\s)+", " ", text)
            # remove extra space at both ends of the text
            text = text.strip()

            # tokenize
            tokenized_text = text.split(" ")
            # remove empty string
            tokenized_text = [w for w in tokenized_text if w]

            # pos tag
            # a list of word tuples # [("great", "JJ"), ("tour", "NN") ...]
            if len(tokenized_text) == 0:
                continue
            word_tuple_list = self.pos_tagger.tag(tokenized_text)

            # remove stop_words
            word_tuple_list = [(w[0].lower(), w[1]) for w in word_tuple_list if w[0].lower() not in self.stopwords]
            # remove empty string
            word_tuple_list = [(w[0], w[1]) for w in word_tuple_list if w[0]]

            combine_or_not = False
            combination_front = ""
            for word_tuple in word_tuple_list:
                # putting them into dictionary
                # add 1 to value if exist
                # add key and value if not
                if word_tuple[1] not in self.pos_tags:
                    if combine_or_not:
                        if combination_front in self.pos_tagged_statistics:
                            self.pos_tagged_statistics[combination_front] += 1
                        else:
                            self.pos_tagged_statistics[combination_front] = 1
                        combine_or_not = False
                        combination_front = ""
                elif word_tuple[0] not in ner_set[review_index]:
                    if combine_or_not:
                        if combination_front:
                            combination_front += "_" + word_tuple[0]
                        else:
                            combination_front = word_tuple[0]
                    else:
                        combine_or_not = True
                        combination_front = word_tuple[0]
            if combine_or_not:
                if combination_front in self.pos_tagged_statistics:
                    self.pos_tagged_statistics[combination_front] += 1
                else:
                    self.pos_tagged_statistics[combination_front] = 1


    def stem(self, candidate_lexicon):
        """ perform stemming on candidate lexicon | candidate lexicon should be a list """

        stemmed_lexicon = []
        for word in candidate_lexicon:
            stemmed_word = self.stemmer.stem(word)
            stemmed_lexicon.append({"word": word, "stemmed_word": stemmed_word})
        stemmed_lexicon = sorted(stemmed_lexicon, key=lambda k: k['word'])

        if self.verbose:
            print "\nMerging stemmed duplicates"
        processed_lexicon = {}
        length = len(stemmed_lexicon)
        cnt = 0
        for word_dict in stemmed_lexicon:
            cnt += 1
            if word_dict["stemmed_word"] not in processed_lexicon:
                processed_lexicon[word_dict["stemmed_word"]] = [word_dict["word"]]
            else:
                processed_lexicon[word_dict["stemmed_word"]].append(word_dict["word"])
            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (cnt, length))
                sys.stdout.flush()

        processed_lexicon = [{"stemmed_word": key, "word": value} for key, value in processed_lexicon.iteritems()]
        # sorting dictionaries by word
        processed_lexicon = sorted(processed_lexicon, key=lambda k: k["stemmed_word"])

        return processed_lexicon


    def load_ner_tags(self, filename):
        """ load named entity for files """

        filename = re.search("([A-Za-z|.]+\-*[A-Za-z|.]+\-*[A-Za-z|.]+\_.*).json", filename).group(1)
        ner_set = []
        with open(os.path.join(self.dst_ne, filename+".txt"), "rb") as ne_f:
            tags = csv.reader(ne_f, delimiter=",")
            for tag in tags:
                if tag[1] == "FILEINDEX":
                    ner_set.append(set())
                else:
                    ner_set[-1].add(tag[0].lower())
        return ner_set


    def render_candidate_lexicon(self):
        """ render the candidate words """

        # filtered by self.frequency_threshold
        if self.verbose:
            print "Filtering out frequency lower than frequency_threshold" + "\n" + "-"*80

        self.create_dir(self.dst)

        pos_tagged_words = []
        pos_tagged_words_under_thre = []
        for key in self.pos_tagged_statistics:
            if self.pos_tagged_statistics[key] > self.frequency_threshold:
                pos_tagged_words.append(key)
            else:
                pos_tagged_words_under_thre.append(key)

        if self.verbose:
            print "Stemming candidate words"
        pos_tagged_words = self.stem(pos_tagged_words)
        pos_tagged_words_under_thre = self.stem(pos_tagged_words_under_thre)

        ordered_dict_list = [[], []]
        if self.verbose:
            print "\nOrganizing candidate words"
        length = len(pos_tagged_words)
        for index in range(len(pos_tagged_words)):
            ordered_dict = OrderedDict()
            ordered_dict["index"] = index + 1
            ordered_dict["count"] = sum([self.pos_tagged_statistics[w] for w in pos_tagged_words[index]["word"]])
            ordered_dict["stemmed_word"] = pos_tagged_words[index]["stemmed_word"]
            ordered_dict["word"] = pos_tagged_words[index]["word"]
            ordered_dict_list[0].append(NoIndent(ordered_dict))

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (index+1, length))
                sys.stdout.flush()

        if self.verbose:
            print "\nOrganizing candidate words <= frequency threshold"
        length = len(pos_tagged_words_under_thre)
        for index in range(len(pos_tagged_words_under_thre)):
            ordered_dict = OrderedDict()
            ordered_dict["index"] = index + 1
            ordered_dict["count"] = sum([self.pos_tagged_statistics[w] for w in pos_tagged_words_under_thre[index]["word"]])
            ordered_dict["stemmed_word"] = pos_tagged_words_under_thre[index]["stemmed_word"]
            ordered_dict["word"] = pos_tagged_words_under_thre[index]["word"]
            ordered_dict_list[1].append(NoIndent(ordered_dict))

            if self.verbose:
                sys.stdout.write("\rStatus: %s / %s" % (index+1, length))
                sys.stdout.flush()

        if self.verbose:
            print "\n" + "-"*80
            print "Saving data to: \033[1m" + self.dst + "\033[0m"
        with open(self.dst, "w+") as f_out:
            f_out.write( json.dumps(ordered_dict_list, indent=4, cls=NoIndentEncoder))


    def create_dir(self, new_path):
        """ create the directory if not exist"""

        dir1 = os.path.dirname(new_path)
        if not os.path.exists(dir1):
            if self.verbose:
                print "Creating directory: " + dir1
                print "-"*80
            os.makedirs(dir1)


    def run(self):
        print "Selecting candidate words" + "\n" + "-"*80

        self.stanford_ner()
        self.get_sentiment_words()
        self.render_candidate_lexicon()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="../data/", help="data source folder")
    parser.add_argument("--freq_thre", type=int, default=100, help="word frequency threshold for candidate sentiment word selection")
    parser.add_argument("--corenlp_path", type=str, default="../stanford-corenlp/", help="the path for Stanford CoreNLP")
    parser.add_argument("--ner_path", type=str, default="../stanford-ner/", help="the path for Stanford NER Tagger")
    parser.add_argument("--verbose", default=False, help="verbose logging or not", action="store_true")
    args = parser.parse_args()

    main = SelectCandidates(src_folder=args.src_folder, freq_thre=args.freq_thre, corenlp_path=args.corenlp_path,\
                            ner_path=args.ner_path, verbose=args.verbose)
    main.run()
