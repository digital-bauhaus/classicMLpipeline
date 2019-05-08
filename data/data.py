import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.df = pd.read_csv(self.corpus_path, delimiter="|", quotechar="\"", header=0)
        self.clean_df()

    def clean_df(self):
        scan = ["class com.github.javaparser.ast.expr.MethodCallExpr"]
        repl = "MethodCallExpression"
        self.df.loc[self.df["CurrentStatementType"].isin(scan), "CurrentStatementType"] = repl
        scan = ["class com.github.javaparser.ast.expr.VariableDeclarationExpr"]
        repl = "VariableDeclarationExpression"
        self.df.loc[self.df["CurrentStatementType"].isin(scan), "CurrentStatementType"] = repl


    def split(self, x, y, test_size=.2):
        """
        Splits given feature-target sets int training and testing sets
        :param x: all features
        :param y: all targets
        :param test_size:
        :return: (train_x, test_x, train_y, test_y)
        """
        return train_test_split(x, y, test_size=test_size, random_state=42, shuffle=True)
