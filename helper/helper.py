from sklearn import metrics
import matplotlib.pyplot as plt
import os

class Helper:

    @staticmethod
    def print_sklearn_classification_report(targets, predictions, classes):
        report = metrics.classification_report(targets, predictions, target_names=classes)
        print(report)


    @staticmethod
    def analyze_results(results):
        print("ANALYSIS:")
        for result in results:
            activated_feature = result[0]
            activated_classifier = result[1]
            print("active feature: %s" % str(activated_feature))
            print("active Classifier: %s" % str(activated_classifier))
            print("accuracy: %.2f" % result[2])
            Helper.print_sklearn_classification_report(result[3][0], result[3][1], result[3][2])
            dest = "confusion_" + result[1].__name__ + ".pdf"
            dest = os.path.join("plots", dest)
            Helper.print_confusion_matrix(result[3][0], result[3][1], result[3][2], result[1].__name__, dest)

        names = [str(feature)+"_"+classifier.__name__ for feature, classifier, _, _ in results]
        heights = [acc for _, _, acc, _ in results]
        names_ix = [i for i, _ in enumerate(names)]
        Helper.bar_plot(names_ix,
                        heights,
                        names,
                        "Repetition Config",
                        "Accuracy",
                        "Experiment Evaluation",
                        "plots/accuracies.pdf")

    @staticmethod
    def bar_plot(x_axis, y_axis, x_ticks, x_label, y_label, title, destination):
        fig = plt.figure()
        plt.tight_layout()
        plt.bar(x_axis, y_axis)
        plt.xticks(x_axis, x_ticks, rotation=90)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()
        plt.show()
        fig.savefig(destination, bbox_inches='tight')

    @staticmethod
    def print_confusion_matrix(targets, predictions, classes, classifier, destination):
        confusion = metrics.confusion_matrix(targets, predictions)
        print(classifier + ":")
        print(confusion)

        # labels = [str(i) for i in range(len(classes))]
        labels = classes

        # plt.tight_layout()
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion)

        for r in range(len(confusion)):
            for c in range(len(confusion[r])):
                ax.text(r,c, confusion[r][c], va="center", ha="center")

        #fig.colorbar(cax)
        ax.set_xticklabels([''] + labels, rotation="vertical")
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion matrix of ' + classifier, pad=180)
        plt.show()
        fig.savefig(destination, bbox_inches='tight', dpi=100)
