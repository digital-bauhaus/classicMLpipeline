from tasks.task import Task
from ml_models.naive_bayes import NaiveBayes
from ml_models.stochastic_gradient_descent import StochasticGradientDescent
from ml_models.support_vector_classifier import SupportVectorClassifier
from ml_models.random_forest import RandomForest
import numpy as np
import itertools


class NumericFeatures(Task):
    def __init__(self, corpus_path, dump_path):
        super(NumericFeatures, self).__init__(corpus_path, dump_path)
        self.features = [
            "NumOfLinesUntilCurrentStatement",
            "NumOfStatementsUntilCurrentStatement",
            "NumOfVariableDeclarationUntilCurrentStatement",
            "NumOfWordsDocString",
            "NumOfMethodParameter"
        ]
        self.classifiers = [
            (NaiveBayes, ["real", self.classes]),
            (StochasticGradientDescent, [self.classes]),
            (SupportVectorClassifier, [self.classes, "linear"]),
            (RandomForest, [self.classes])
        ]

    def single_feature(self):
        results = []
        x = self.data.df[self.features]
        y = self.data.df["CurrentStatementType"]
        train_x, test_x, train_y, test_y = self.data.split(x, y)

        for feature in self.features:
            feature_x = np.array(train_x[feature]).reshape(-1, 1)
            feature_y = np.array(self.__y_to_one_hot__(train_y))

            feature_x_test = np.array(test_x[feature]).reshape(-1, 1)
            feature_y_test = np.array(self.__y_to_one_hot__(test_y))

            for Classifier, params in self.classifiers:
                c = Classifier(*params)
                c.fit(feature_x, feature_y)
                print(feature + " " + str(Classifier))
                acc = c.test(feature_x_test, feature_y_test, False)
                #print("accuracy: %.4f" % acc)
                results.append([feature, Classifier, acc, [feature_y_test, c.predicted, self.classes]])
        return results

    def pairwise_feature(self):
        results = []
        x = self.data.df[self.features]
        y = self.data.df["CurrentStatementType"]
        train_x, test_x, train_y, test_y = self.data.split(x, y)

        combinations = itertools.combinations(self.features, 2)
        for combination in combinations:
            comb = list(combination)
            feature_x = np.array(train_x[comb])
            feature_y = np.array(self.__y_to_one_hot__(train_y))

            feature_x_test = np.array(test_x[comb])
            feature_y_test = np.array(self.__y_to_one_hot__(test_y))

            for Classifier, params in self.classifiers:
                c = Classifier(*params)
                c.fit(feature_x, feature_y)
                print(str(comb) + " " + str(Classifier))
                acc = c.test(feature_x_test, feature_y_test, False)
                #print("classifier: %s - accuracy: %.4f" % (str(Classifier), acc))
                results.append([combination, Classifier, acc, [feature_y_test, c.predicted, self.classes]])
        return results

    def all_features(self):
        results = []
        x = self.data.df[self.features]
        y = self.data.df["CurrentStatementType"]
        train_x, test_x, train_y, test_y = self.data.split(x, y)
        feature_x = np.array(train_x)
        feature_y = np.array(self.__y_to_one_hot__(train_y))
        feature_x_test = np.array(test_x)
        feature_y_test = np.array(self.__y_to_one_hot__(test_y))
        for Classifier, params in self.classifiers:
            print("all " + str(Classifier))
            c = Classifier(*params)
            c.fit(feature_x, feature_y)
            acc = c.test(feature_x_test, feature_y_test, False)
            #print("classifier: %s - accuracy: %.4f" % (str(Classifier), acc))
            #print("#####################################")
            results.append(["all", Classifier, acc, [feature_y_test, c.predicted, self.classes]])
        return results