# UGSD: User Generated Sentiment Dictionaries from Online Customer Reviews
## 1. Introduction
The UGSD toolkit is a representation learning framework designed to generate domain-specific sentiment dictionaries from online customer reviews ([UGSD: User Generated Sentiment Dictionaries from Online Customer Reviews]()). For example, given reviews on TripAdvisor, our framework attempts to generate sentiment lexicons of different ratings from 5-star to 1-star, which contain words conveying positive or negative polarities on attractions or tours respectively.

In specific, the UGSD toolkit can generate sentiment dictionaries of different domains given the reviews and the associated ratings in a specific domain. Especially, the lexicons from the UGSD toolkit include not only sentiment words but also representations of them, which can be used in other advanced applications such as sentiment analysis or dictionary expansion.

### 1.1. Requirements
- python2.7
- Stanford CoreNLP
- Stanford NER software
- proNet

### 1.2. Getting Started
#### Download:
```
$ git clone https://github.com/cnclabs/UGSD
$ cd ./UGSD
```

#### Download Stanford CoreNLP:
The UGSD toolkit uses Stanford CoreNLP to tag part-of-speech of words. So, we need to download the Stanford CoreNLP software from [Stanford CoreNLP software official website](https://stanfordnlp.github.io/CoreNLP/).  
Then, users need to rename Stanford CoreNLP directory to be \"stanford-corenlp\" and move it in this repository, or just add the Stanford CoreNLP software path when users run programs.
```
$ mv stanford-corenlp-full-2018-XX-XX stanford-corenlp
```

#### Download Stanford NER:
The UGSD toolkit uses Stanford NER to recognize named entity. Therefore, we need to download the Stanford NER software from [Stanford NER software official website](https://nlp.stanford.edu/software/CRF-NER.html).  
Then, again, users need to rename Stanford NER directory to be \"stanford-ner\" and move it in this repository, or just add the Stanford NER software path when users run programs.
```
$ mv stanford-ner-2018-XX-XX stanford-ner
```

#### Download proNet
The UGSD toolkit uses proNet to model the direct proximity of nodes in a network in order to learn the relationships between sentiment words and rating symbols. So, we need to download this package and compile them. Users can refer to [proNet github repository](https://github.com/cnclabs/proNet-core) for more details about the usage of this framework.
```
$ git clone https://github.com/cnclabs/proNet-core.git
$ cd proNet-core
$ make
$ cd ..
```

#### Install python packages:
```
$ pip install -r requirements.txt
```

#### Prepare reviews
Users need to prepare reviews in a specified input format. At first, users specified a source folder, which is called `./data/` as the default path in programs. Then users will put all reviews in the path `./data/reviews/` in the JSON format, which are named as `Amsterdam_01.json` or `Paris_05.json` for example. Review files include the category, the rated entity and associated reviews. For each of reviews, there are information about the index, the rating and the corresponding review text.  
Here is an example for the review format in all review files.
```
{
    "category": "Amsterdam",
    "entity": "Room-Escape-Games",
    "reviews": [
        {
            "index": 1,
            "rating": 5,
            "review": "Fun! ..."
        },
        {
            "index": 2,
            "rating": 5,
            "review": "Good place! ..."
        },
        {
            "index": 3,
            "rating": 4,
            "review": "Nice! ..."
        }
    ]
}
```

Then, after downloading Stanford CoreNLP, Stanford NER software and proNet, the default tree structure for this repository is listed as the following. Otherwise, users need to specify the needed path for tools when they run programs.
```
.
|-- data/
    |-- reviews/
        |-- Amsterdam_01.json
        |-- Amsterdam_02.json
        |-- Paris_01.json
        |-- Paris_02.json
|-- proNet-core/
|-- README.md
|-- requirement.txt
|-- stanford-corenlp/
|-- stanford-ner/
|-- UGSD/
```

## 2. Usages
Users can run the shell script for the whole procedures.
```
$ cd UGSD
$ sh main.sh
```
However, users are recommended to run each program respectively because the steps for candidate sentiment word selection and review transformation take a long time. Then, we explain the usage and the parameters of each step accordingly.

### 2.1. Candidate Sentiment Word Selection
Our framework selects candidate sentiment words from all reviews by leveraging POS information.

#### Run
```
$ python SelectCandidates.py [--src_folder <string>] [--freq_thre <int>] [--corenlp_path <string>] [--ner_path <string>] [--verbose]
```

#### Parameters
```
    --src_folder <string>
        data source folder
    --freq_thre <int>
        word frequency threshold for candidate sentiment word selection
    --corenlp_path <string>
        path for Stanford CoreNLP
    --ner_path <string>
        path for Stanford NER tagger
    --verbose
        verbose logging or not
```

### 2.2 Review Transformation
In this step, our framework substitutes all entities in reviews with associated ratings.

#### Run
```
$ python SubstituteEntity.py --src ../data/reviews/Amsterdam_01.json [--src_folder <string>] [--corenlp_path <string>] [--verbose]
```

#### Parameters
```
    --src <string>
        review path
    --src_folder <string>
        data source folder
    --corenlp_path <string>
        path for Stanford CoreNLP
    --verbose
        verbose logging or not
```

If users need to have custom ways to recognize entities, they can write their own functions about the regular expression generation in accordance with reviews in different domains. And, they can pass the function to the constructor of the class in the program, `SubstituteEntity.py`. In the sample code, the program aims to recognize entities in reviews from TripAdvisor.

### 2.3 Reviews Grouping
Our framework merges all processed reviews into a file for the next step, network construction.

#### Run
```
$ python MergeCorpus.py [--src_folder <string>] [--verbose]
```

#### Parameters
```
    --src_folder <string>
        data source folder
    --verbose
        verbose logging or not
```

### 2.4 Co-occurrence Network Construction
In this step, our framework constructs co-occurrence networks, which will be used for the representation learning.

#### Run
```
$ python ConstructNetwork.py [--src_folder <string>] [--min_count <int>] [--window_size <int>] [--vocab_path <string>] [--cooccur_path <string>] [--auto_window] [--verbose]
```

#### Parameters
```
    --src_folder <string>
        data source folder
    --min_count <int>
        words with less than the minimum count are excluded
    --window_size <int>
        window size for constructing networks
    --vocab_path <string>
        path for vocabularies in the network
    --cooccur_path <string>
        path for co-occurrence networks
    --auto_window
        automatically calculate the window size or not
    --verbose
        verbose logging or not
```

### 2.5 Representation Learning
Users use proNet to model the direct proximity of nodes in the network in order to learn low-dimensional representations. For more details of this package, please refer to the [proNet](https://github.com/cnclabs/proNet-core).

#### Run
```
$ ../proNet-core/cli/line -train ../data/network/cooccur/starred.txt -save ../data/network/cooccur/starred_repr.txt -order 1 -dimensions 200 -sample_times 25 -negative_samples 5 -threads 2
```

### 2.6 Dictionary Construction
With the learned vector representations of words and rating symbols, our framework constructs dictionaries of all ratings by the maximum-cosine-similarity scheme and the z-score scheme. The generated dictionaries will be in the `src_folder/starred_lexicons/`.

#### Run
```
$ python ConstructLexicon.py [--src_folder <string>] [--min_star <int>] [--max_star <int>] [--star_scale <int>] [--std_threshold <float>] [--verbose]
```

#### Parameters
```
    --src_folder <string>
        data source folder
    --min_star <int>
        minimum rating scale
    --max_star <int>
        maximum rating scale
    --star_scale <int>
        rating scale
    --std_threshold <float>
        the threshold for the z-score scheme
    --verbose
        verbose logging or not
```

## 3. Citation
```
@inproceedings{,
    author = {Wang, Chun-Hsiang and Fan, Kang-Chun and Wang, Chuan-Ju and Tsai, Ming-Feng},
    title = {UGSD: User Generated Sentiment Dictionaries from Online Customer Reviews},
    booktitle = {},
    series = {},
    year = {2018},
    isbn = {},
    location = {},
    pages = {},
    numpages = {10},
    url = {http://doi.acm.org/1},
    doi = {},
    acmid = {},
    publisher = {ACM},
    address = {New York, NY, USA},
    keywords = {sentiment analysis, dictionary construction, user-generated content, representation learning, opinion mining},
} 
```

