import pandas as pd
from imblearn.over_sampling import RandomOverSampler

input_csv_file = "five_fold/train_fold_5.csv"
output_csv_file = "five_fold/train_fold_5_relevance_upsampled.csv"
column_name = "relevance"

df = pd.read_csv(input_csv_file)

X = df.drop(columns=[column_name])
y = df[column_name]

oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

upsampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=column_name)], axis=1)

upsampled_df.to_csv(output_csv_file, index=False)

print("Upsampled data saved to", output_csv_file)