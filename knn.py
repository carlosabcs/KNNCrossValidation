import pandas as pd
import numpy as np


class KNNClassifier():
    def __init__(self, nn = 3):
        self.nn = nn
        self.accuracy = None
        self.accuracy_std = None
        self.f1_score = None
        self.f1_score_std = None


    def get_folds(self, data, k):
        # Get subsets based on the target class
        positive = data[data['Outcome'] == 1]
        negative = data[data['Outcome'] == 0]

        # Split the subsets into stratified sub-subsets
        positive_folds = np.split(
            positive.sample(frac=1),
            [ int((1 / k) * i * len(positive)) for i in range(1, k) ]
        )
        negative_folds = np.split(
            negative.sample(frac=1),
            [ int((1 / k) * i * len(negative)) for i in range(1, k) ]
        )
        # Concat stratified subsets to create stratified sets
        folds = []
        for i in range(k):
            folds.append(
                pd.concat([
                    positive_folds[i],
                    negative_folds[i]
                ])
            )
        return folds


    def knn_predict(self, nn, element, train_data):
        train_data['euclidean_distance'] = (
            ((element['Pregnancies'] - train_data['Pregnancies']) ** 2) +
            ((element['Glucose'] - train_data['Glucose']) ** 2) +
            ((element['BloodPressure'] - train_data['BloodPressure']) ** 2) +
            ((element['SkinThickness'] - train_data['SkinThickness']) ** 2) +
            ((element['Insulin'] - train_data['Insulin']) ** 2) +
            ((element['BMI'] - train_data['BMI']) ** 2) +
            ((element['DiabetesPedigreeFunction'] - train_data['DiabetesPedigreeFunction']) ** 2) +
            ((element['Age'] - train_data['Age']) ** 2)
        ) ** 0.5

        sorted_data = train_data.sort_values(by='euclidean_distance')

        votes = [0, 0]
        for _, row in sorted_data.head(nn).iterrows():
            votes[int(row['Outcome'])] += 1
        return 0 if votes[0] > votes[1] else 1


    def calculate_accuracy(self, tp, tn, fp, fn):
        n = tp + tn + fp + fn
        return (tp + tn) / n


    def calculate_f1_score(self, tp, tn, fp, fn, beta = 1):
        prec = (tp / (tp + fp))
        rec = (tp / (tp + fn))
        return (
            (1 + (beta ** 2)) *\
            (
                (prec * rec) /\
                (((beta ** 2) * prec) + rec)
            )
        )


    def cross_validate(
        self,
        data,
        k_folds,
        r = 1
    ):
        print('===== KNN with K = %s =====' % self.nn)
        global_acc_list = []
        global_f1_list = []
        for it in range(r):
            if r > 1:
                print('ITERATION %s:' % (it + 1))
            folds = self.get_folds(data, k_folds)
            acc_list = []
            f1_list = []
            for i in range(len(folds)):
                # Train data is composed by all folds except the current one
                train_data = pd.concat(folds[:i] + folds[i+1:])
                # Test data is composed by current fold
                test_data = folds[i]
                true_positives = 0
                true_negatives = 0
                false_positives = 0
                false_negatives = 0
                for _, row in test_data.iterrows():
                    predicted = self.knn_predict(self.nn, row, train_data)
                    if row['Outcome'] == 1:
                        if predicted == row['Outcome']:
                            true_positives += 1
                        else:
                            false_negatives += 1
                    else:
                        if predicted == row['Outcome']:
                            true_negatives += 1
                        else:
                            false_positives += 1

                accuracy = self.calculate_accuracy(
                    true_positives,
                    true_negatives,
                    false_positives,
                    false_negatives
                )
                f1 = self.calculate_f1_score(
                    true_positives,
                    true_negatives,
                    false_positives,
                    false_negatives
                )
                acc_list.append(accuracy)
                f1_list.append(f1)
                global_acc_list.append(accuracy)
                global_f1_list.append(f1)
                if r == 1:
                    print('Fold %s: acc(%.3f), f1(%.3f)' % (
                        i + 1, accuracy, f1
                    ))
            print('Average accuracy: %.3f (%.3f)' % (
                np.mean(acc_list), np.std(acc_list)
            ))
            print('Average f1 score: %.3f (%.3f)\n' % (
                np.mean(f1_list), np.std(f1_list)
            ))
        self.accuracy = np.mean(global_acc_list)
        self.accuracy_std = np.std(global_acc_list)
        self.f1_score = np.mean(global_f1_list)
        self.f1_score_std = np.std(global_f1_list)
