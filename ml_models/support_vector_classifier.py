from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from helper.helper import Helper
from .base import Base

class SupportVectorClassifier(Base):
    def __init__(self, classes, t="linear", use_text_feature=False):
        """
        Constructor
        multinomial to use MultinomialNB for text classification
        real to use GaussianNB for real valued data
        :param model_type: [multinomial, real]
        :param classes:
        """
        super(SupportVectorClassifier, self).__init__(classes)
        self.use_text_feature = use_text_feature
        if t == "linear":
            self.Classifier = LinearSVC
            self.c = self.Classifier()
        else:
            self.Classifier = SVC
            self.c = self.Classifier()
            self.pipe = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SVC())
            ])

    def fit(self, x, y):
        if not self.use_text_feature:
            super(SupportVectorClassifier, self).fit(x, y)
            self.trained = True
        else:
            self.pipe.fit(x, y)
            self.trained = True

    def predict(self, x):
        if not self.use_text_feature:
            return super(SupportVectorClassifier, self).predict(x)
        else:
            return self.pipe.predict(x)

    def test(self, x, y, print_report=False):
        if not self.use_text_feature:
            return super(SupportVectorClassifier, self).test(x, y, print_report)
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