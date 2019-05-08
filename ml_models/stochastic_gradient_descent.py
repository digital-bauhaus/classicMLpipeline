from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from helper.helper import Helper
from .base import Base

class StochasticGradientDescent(Base):
    def __init__(self,
                 classes,
                 use_text_feature=False,
                 loss="hinge",
                 penalty="l2",
                 alpha=1e-3,
                 random_state=42,
                 max_iter=20,
                 tol=None):
        """
        Constructor
        :param classes:
        """
        super(StochasticGradientDescent, self).__init__(classes)
        self.use_text_feature = use_text_feature
        self.Classifier = SGDClassifier
        self.c = self.Classifier(loss=loss,
                                 penalty=penalty,
                                 alpha=alpha,
                                 random_state=random_state,
                                 max_iter=max_iter,
                                 tol=tol)
        self.pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.c)
        ])

    def fit(self, x, y):
        if not self.use_text_feature:
            super(StochasticGradientDescent, self).fit(x, y)
            self.trained = True
        else:
            self.pipe.fit(x, y)
            self.trained = True

    def predict(self, x):
        if not self.use_text_feature:
            return super(StochasticGradientDescent, self).predict(x)
        else:
            return self.pipe.predict(x)

    def test(self, x, y, print_report=False):
        if not self.use_text_feature:
            return super(StochasticGradientDescent, self).test(x, y, print_report)
        else:
            assert self.trained
            predicted = self.predict(x)
            test_array = []
            for i in range(len(predicted)):
                if predicted[i] == y[i]:
                    test_array.append(True)
                else:
                    test_array.append(False)
            self.predicted = predicted
            self.tested = True
            if print_report:
                Helper.print_sklearn_classification_report(y, predicted, self.classes)
            return sum(test_array)/len(test_array)
