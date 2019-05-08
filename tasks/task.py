import pandas as pd
import pickle
import os
from data.data import Data


class Task:
    """
    Task base class.
    Provides functions for all concrete tasks in common.
    Holds the data and the classes
    """
    def __init__(self, corpus_path, dump_path):
        self.data = Data(corpus_path)
        self.classes = self.__get_classes__("CurrentStatementType")
        self.dump_path = dump_path
        self.store_classes()

    def __get_classes__(self, df_target_column_name):
        """
        Get the classes for the task
        :param df_target_column_name: name of the target column holding the classes
        :return: list of all classes
        """
        result = list(set(list(self.data.df[df_target_column_name])))
        return result

    def __y_to_one_hot__(self, y):
        """
        translates a list of target strings to a list of indices corresponding to their index in self.classes
        :param y: list of target strings
        :return: list of target indices
        """
        result = []
        classes = {}
        for index, classname in enumerate(self.classes):
            classes[classname] = index

        for item in y:
            result.append(classes[item])
        return result

    def store_classes(self):
        with open(os.path.join(self.dump_path, "classes.pkl"), "wb") as f:
            pickle.dump(self.classes, f)