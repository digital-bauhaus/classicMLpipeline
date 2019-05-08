from helper.helper import Helper
import pickle
import os
class Base:
    def __init__(self, classes):
        self.classes = classes
        self.trained = False
        self.tested = False
        self.test_array = []
        self.predicted = []

        self.dump_path = ""

    def fit(self, x, y):
        """
        train model with training data
        :param x: dim: [rows, features]
        :param y: dim: [rows]
        :return:
        """
        self.c.fit(x, y)

    def test(self, x, y, print_report=False):
        assert self.trained
        predicted = self.predict(x)
        self.test_array = predicted == y
        self.predicted = predicted
        self.tested = True
        if print_report:
            Helper.print_sklearn_classification_report(y, predicted, self.classes)
        return sum(self.test_array)/len(self.test_array)

    def predict(self, x):
        return self.c.predict(x)

    # def get_vocab(self):
    #     assert self.pipe is not None
    #     count_vect = self.pipe.get_params()["vect"]
    #     vocab = count_vect.vocabulary_
    #     return vocab

    # def get_idf(self):
    #     assert self.pipe is not None
    #     tfidf = self.pipe.get_params()["tfidf"]
    #     idf = tfidf.idf_
    #     return idf
    #
    # def set_vocabidf(self):
    #     assert self.vocab is not None and self.idf is not None
    #     self.pipe.get_params()["vect"].vocabulary = self.vocab
    #     self.pipe.get_params()["tfidf"].idf_ = self.idf

    def store(self, name):
        if not self.trained:
            return

        classifier_fqn = os.path.join(self.dump_path, name + "_classifier.pkl")
        # vocab_fqn = os.path.join(self.dump_path, name + "_vocab.pkl")
        # idf_fqn = os.path.join(self.dump_path, name + "_idf.pkl")
        with open(classifier_fqn, "wb") as f:
            pickle.dump(self.pipe, f)
        # with open(vocab_fqn, "wb") as f:
        #     pickle.dump(self.get_vocab(), f)
        # with open(idf_fqn, "wb") as f:
        #     pickle.dump(self.get_idf(), f)

    def load(self, path, name):
        classifier_fqn = os.path.join(path, name + "_classifier.pkl")
        # vocab_fqn = os.path.join(path, name + "_vocab.pkl")
        # idf_fqn = os.path.join(path, name + "_idf.pkl")
        with open(classifier_fqn, "rb") as f:
            self.pipe = pickle.load(f)
        # with open(vocab_fqn, "rb") as f:
        #     self.vocab = pickle.load(f)
        # with open(idf_fqn, "rb") as f:
        #     self.idf = pickle.load(f)
