from helper.helper import Helper
from tasks.text_feature import TextFeatures


def run(corpus_path):
    experiment = TextFeatures(corpus_path, "")
    result = experiment.text_only_feature()
    Helper.analyze_results(result)
