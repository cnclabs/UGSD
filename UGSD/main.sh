# candidate sentiment word selection
python SelectCandidates.py --verbose

# review transformation
for i in ../data/reviews/*.json
do
    python SubstituteEntity.py --src $i --verbose
done

# corpus grouping
python MergeCorpus.py --verbose

# co-occurrence network construction
python ConstructNetwork.py --verbose

# representation learning
../proNet-core/cli/line -train ../data/network/cooccur/starred.txt -save ../data/network/cooccur/starred_repr.txt -order 1 -dimensions 200 -sample_times 25 -negative_samples 5 -threads 2

# dictionary construction
python ConstructLexicon.py --verbose
