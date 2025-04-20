import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

train_file = "file.csv"
val_file = "file.csv"
test_file = "file.csv"
attribute = "relevance"


train_data = pd.read_csv(train_file)
validation_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)

def merge_patch_msg(df):
    def merge_msg_patch(row):
        return row['msg'] + ' [SEP] ' + row['patch']

    df['merged'] = df.apply(merge_msg_patch, axis=1)


    return df

if attribute == "informativeness" or attribute == "expression":
    X_train = train_data["msg"]
    y_train = train_data[attribute]

    X_validation = validation_data["msg"]
    y_validation = validation_data[attribute]

    X_test = test_data["msg"]
    y_test = test_data[attribute]
else:
    merged_x_train = merge_patch_msg(train_data)
    merged_x_val = merge_patch_msg(validation_data)
    merged_x_test = merge_patch_msg(test_data)

    X_train = merged_x_train["merged"]
    y_train = train_data[attribute]

    X_validation = merged_x_val["merged"]
    y_validation = validation_data[attribute]

    X_test = merged_x_test ["merged"]
    y_test = test_data[attribute]


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)
X_test_tfidf = vectorizer.transform(X_test)


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2024)
rf_classifier.fit(X_train_tfidf, y_train)


train_predictions = rf_classifier.predict(X_train_tfidf)
validation_predictions = rf_classifier.predict(X_validation_tfidf)
test_predictions = rf_classifier.predict(X_test_tfidf)


test_accuracy = accuracy_score(y_test, test_predictions)
balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)


print("Accuracy:", test_accuracy)
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')