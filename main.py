import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from knn import KNNClassifier


def normalize(data, output_col):
    for column in data.columns:
        if column == output_col:
            continue

        max_val = data[column].max()
        min_val = data[column].min()
        data[column] = data[column].apply(
            lambda x: (x - min_val) / (max_val - min_val)
        )
    return data


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_results(results):
    x_labels = list(sorted(results.keys()))
    y_accuracy = []
    y_f1 = []
    for label in x_labels:
        y_accuracy.append(results[label][0])
        y_f1.append(results[label][1])

    plt.rcParams.update({'font.size': 12})
    bar_width = .25

    x = np.arange(len(x_labels))
    _, ax = plt.subplots()
    r1 = ax.bar(x - bar_width / 2, y_accuracy, bar_width, color='#fdbf11', label='Accuracy')
    r2 = ax.bar(x + bar_width / 2, y_f1, bar_width, color='#000000', label='F1 score')
    ax.set_title('Scores obtained vs the number of nearest neighbors in KNN')
    ax.set_ylabel('Scores')
    ax.set_xlabel('# nearest neighbors')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    autolabel(r1, ax)
    autolabel(r2, ax)
    plt.show()


def main():
    df = pd.read_csv('./diabetes.csv')
    normalized_data = normalize(df, 'Outcome')
    knn_classifier = KNNClassifier()
    for k in [5, 10]:
        results = {}
        for nn in [3, 5, 7]:
            knn_classifier.nn = nn
            knn_classifier.cross_validate(
                normalized_data,
                k, # k folds
                1 # r
            )
            results[str(nn)]= [
                knn_classifier.accuracy,
                knn_classifier.f1_score
            ]
        plot_results(results)


    for nn in [3, 5, 7]:
        knn_classifier.nn = nn
        knn_classifier.cross_validate(
            normalized_data,
            10, # k folds
            10 # r
        )
        print('\nGlobal accuracy: %.3f (%.3f)' % (
            knn_classifier.accuracy,
            knn_classifier.accuracy_std
        ))
        print('Global f1 score accuracy: %.3f (%.3f)\n' % (
            knn_classifier.f1_score,
            knn_classifier.f1_score_std
        ))
        results[str(nn)]= [
            knn_classifier.accuracy,
            knn_classifier.f1_score
        ]
    plot_results(results)


if __name__ == "__main__":
    main()