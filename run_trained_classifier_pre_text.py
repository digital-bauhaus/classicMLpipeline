import sys
import pickle
import os
from ml_models.random_forest import RandomForest
from ml_models.naive_bayes import NaiveBayes
def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    args = sys.argv
    feature = args[1]
    classifier = args[2]
    text = [args[3]]
    #print(text)
    classes = load_classes()
    #print(classes)
    if classifier == "RF":
        c = RandomForest(classes, True)
        class_name = feature + "_" + str(RandomForest.__name__)
        c.load("", class_name)
        predicted = c.predict(text)
        print(classes[predicted[0]])
    elif classifier == "NB":
        c = NaiveBayes("multinomial", classes, True)
        class_name = feature + "_" + str(NaiveBayes.__name__)
        c.load("", class_name)
        predicted = c.predict(text)
        print(classes[predicted[0]])



def load_classes():
    classes_path = "classes.pkl"
    with open(classes_path, "rb") as f:
        classes = pickle.load(f)
    return classes


if __name__ == "__main__":
    main()
