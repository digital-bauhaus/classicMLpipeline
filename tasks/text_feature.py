from ml_models.naive_bayes import NaiveBayes
from ml_models.random_forest import RandomForest
from ml_models.stochastic_gradient_descent import StochasticGradientDescent
from ml_models.support_vector_classifier import SupportVectorClassifier
from tasks.task import Task
import numpy as np
import time

class TextFeatures(Task):
    """
    Task for text features only.
    Uses the text
    """
    def __init__(self, corpus_path, dump_path):
        super(TextFeatures, self).__init__(corpus_path, dump_path)
        self.features = [
            #"CurrentStatementAsSourceCode",
            "SourceCodeUntilCurrentStatement"
        ]
        self.classifiers = [
            (NaiveBayes, ["multinomial", self.classes, True]),
            (StochasticGradientDescent, [self.classes, True]),
            (SupportVectorClassifier, [self.classes, None, True]),
            (RandomForest, [self.classes, True])
        ]
        self.dump_path = ""

    def text_only_feature(self):
        results = []
        x = self.data.df[self.features]
        y = self.data.df["CurrentStatementType"]
        train_x, test_x, train_y, test_y = self.data.split(x, y)

        train_x = [item[0] for item in train_x.values.tolist()]
        test_x = [item[0] for item in test_x.values.tolist()]

        train_y = self.__y_to_one_hot__(train_y)
        test_y = self.__y_to_one_hot__(test_y)

        for feature in self.features:
            for Classifier, params in self.classifiers:
                start = time.time()
                c = Classifier(*params)
                print(feature + " " + str(Classifier))
                c.fit(train_x, train_y)
                acc = c.test(test_x, np.array(test_y), False)
                # print("accuracy: %.4f" % acc)
                results.append([feature, Classifier, acc, [test_y, c.predicted, self.classes]])
                end = time.time()
                duration = end - start
                print("run took: %s" % (str(time.strftime("%H:%M:%S", time.gmtime(duration)))))
                c.store(feature + "_" + str(Classifier.__name__))
        return results
