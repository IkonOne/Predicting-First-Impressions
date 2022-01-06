import numpy as np
import matplotlib.pyplot as plt
from pandas.core.algorithms import isin
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import os
os.makedirs('plots', exist_ok=True)

def prep_data_for_violin(train, test, name):
    """Prepares a Pandas DataFrame of the data to be used with a Seaborn violin plot."""
    if not isinstance(train, (list, tuple)):
        train = [train]
        test = [test]
        name = [name]

    df = pd.DataFrame(columns=["values", "names", "types"])
    for i in range(len(train)):
        df_train = pd.DataFrame(
            {
                "values": train[i],
                "names": [name[i]] * len(train[i]),
                "types": ["train"] * len(train[i])
            }
        )

        df_test = pd.DataFrame(
            {
                "values": test[i],
                "names": [name[i]] * len(test[i]),
                "types": ["test"] * len(test[i])
            }
        )
        df = pd.concat([df, df_train, df_test])
    
    return df

def plot_violin_comparison(train, test, title, filename):
    df = prep_data_for_violin(train,test,title)
    sns.violinplot(data=df, x="names", y="values", hue="types", split=True)

    plt.savefig(f'plots/{filename}.png')
    plt.close()

def main():
    import data_prep

    label_idx=3
    X, y, labels, img_names = data_prep.load_cleaned_data()
    y = y[:,label_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    y_pred = np.mean(y_train)
    y_center = np.abs(y_train - y_pred)
    y_err = np.abs(y_test - y_pred)

    plot_violin_comparison(y_center, y_err, f'Guess the Mean ({labels[label_idx]})', f'gtm_{labels[label_idx]}')

    model = keras.models.load_model('./Models/Trustworthiness.h5')
    X_train = np.expand_dims(X_train, axis=-1)
    y_train_pred = model.predict(X_train)
    X_test = np.expand_dims(X_test, axis=-1)
    y_test_pred = model.predict(X_test)

    y_train_err = np.abs(np.expand_dims(y_train, axis=-1) - y_train_pred)
    y_train_err = y_train_err.reshape((y_train_err.shape[0]))
    y_test_err = np.abs(np.expand_dims(y_test, axis=-1) - y_test_pred)
    y_test_err = y_test_err.reshape((y_test_err.shape[0]))

    plot_violin_comparison(y_train_err, y_test_err, f'Prediction First Impressions (Trustworthiness)', f'pfi_Trustworthiness')

    plot_violin_comparison(
        [y_center, y_train_err, y_train_err],
        [y_err, y_test_err, y_train_err],
        ["gtm", "pfi", "pfi2"],
        "gtm_vs_pfi"
    )

if __name__ == '__main__':
    main()