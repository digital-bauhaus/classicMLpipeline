import experiments.numeric_feature_experiment as nfe
import experiments.text_feature_experiment as tfe
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    #corpus_path = "/mnt/media/Corpora/AndreKarge_2019-04-03_NumericFeaturesCallVSVarDeclStatement/corpus.csv"
    corpus_path = "/mnt/media/foo.csv"
    #nfe.run(corpus_path)
    tfe.run(corpus_path)