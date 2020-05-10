import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def prepare_train_test_data(n_samples: int):
    raw_dataset = pd.read_csv('data/ru_train.csv')

    raw_dataset['before'] = raw_dataset['before'].astype(str)
    raw_dataset['after'] = raw_dataset['after'].astype(str)

    d = raw_dataset['class'].value_counts().to_dict()

    for i in d:
        if d[i] > n_samples:
            d[i] = n_samples

    rus = RandomUnderSampler(sampling_strategy=d, random_state=0)

    raw_dataset_resampled, _ = rus.fit_resample(raw_dataset, raw_dataset['class'])

    df = raw_dataset_resampled[['before', 'after']]

    df_train, df_test = train_test_split(df, test_size=0.1)

    df_train.to_csv('train_dataset.tsv', index=False, header=False, sep='\t')
    df_test.to_csv('test_dataset.tsv', index=False, header=False, sep='\t')
