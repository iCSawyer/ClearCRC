import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data


def calculate_metrics(data, attribute):
    y_true = []
    y_pred = []
    for row in data:
        if "1" not in str(row[attribute + '_ref']) and "0" not in str(row[attribute + '_gen']):
            y_true.append(0)
        else:
            y_true.append(int(row[attribute + '_ref']))
        if "1" not in str(row[attribute + '_gen']) and "0" not in str(row[attribute + '_gen']):
            y_pred.append(0)
        else:
            y_pred.append(int(row[attribute + '_gen']))

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, balanced_accuracy, precision, recall, f1


def main(file_patch, attribute):
    data = read_csv_file(file_path)
    accuracy, balanced_accuracy, precision, recall, f1 = calculate_metrics(data, attribute)

    print(f'Accuracy: {accuracy:.3f}')
    print(f'Balanced Accuracy: {balanced_accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')


if __name__ == '__main__':
    file_path = 'result_test_fold_1.csv'
    print(file_path)
    print("------------------------------")
    attribute = 'relevance'
    print(attribute)
    main(file_path, attribute)
    print("------------------------------")
    attribute = 'informativeness'
    print(attribute)
    main(file_path, attribute)
    print("------------------------------")
    attribute = 'expression'
    print(attribute)
    main(file_path, attribute)