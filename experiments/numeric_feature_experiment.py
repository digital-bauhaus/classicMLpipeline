from helper.helper import Helper
from tasks.numeric_features import NumericFeatures


def run(corpus_path):
    experiment = NumericFeatures(corpus_path)
    results = []
    print("----------- single feature ----------")
    results += experiment.single_feature()
    print("-------------- pairwise -------------")
    results += experiment.pairwise_feature()
    print("---------------- all ----------------")
    results += experiment.all_features()

    Helper.analyze_results(results)