from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from helper.helper import Helper
from .base import Base
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class NaiveBayes(Base):
    def __init__(self, model_type, classes, use_text_feature=False):
        """
        Constructor
        multinomial to use MultinomialNB for text classification
        real to use GaussianNB for real valued data
        :param model_type: [multinomial, real]
        :param classes:
        """
        super(NaiveBayes, self).__init__(classes)
        self.model_type = model_type
        self.use_text_feature = use_text_feature
        if self.model_type == "multinomial":
            self.Classifier = MultinomialNB
        elif self.model_type == "real":
            self.Classifier = GaussianNB
        self.c = self.Classifier()
        if use_text_feature:
            self.pipe = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
            ])

    def fit(self, x, y):
        if not self.use_text_feature:
            super(NaiveBayes, self).fit(x, y)
            self.trained = True
        else:
            self.pipe.fit(x, y)
            self.trained = True

    def predict(self, x):
        if not self.use_text_feature:
            return super(NaiveBayes, self).predict(x)
        else:
            return self.pipe.predict(x)

    def test(self, x, y, print_report=False):
        if not self.use_text_feature:
            return super(NaiveBayes, self).test(x, y, print_report)
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


