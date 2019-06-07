from .base import Base
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from helper.helper import Helper
from helper.Tokenizer import Tokenizer as SpinellisTokenizer

class TokenizerTest(Base):
    def __init__(self, classes, n_estimators=500, n_jobs=6):
        super(TokenizerTest, self).__init__(classes)
        self.dump_path = "test"
        self.tokenizer = SpinellisTokenizer()
        self.pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenizer,
                                     strip_accents=None,
                                     lowercase=False)),  # vectorizer
            ('tfidf', TfidfTransformer()),  # transformer
            ('clf', RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs))
        ])


    def fit(self, x, y):
        #_x = [x[0]]
        #_y = [y[0]]
        self.pipe.fit(x, y)
        self.trained = True

    def predict(self, x):
        return self.pipe.predict(x)

    def test(self, x, y, print_report=False):
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
        return sum(test_array) / len(test_array)
