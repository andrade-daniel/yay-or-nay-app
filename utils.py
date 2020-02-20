import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,

)
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true, y_pred, normalize=False, classes=None, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, classes)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    return ax


class DropCols(object):
    '''
    Class to drop unwanted columns
    '''
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def _delSameCols(self, cols_to_drop):
        cols = self.cols_to_drop
        cols = list(set(cols))
        # print(u"      - %s features to be removed" % len(cols))
        return cols

    def transform(self, X):
        dat = X.copy()
        lstcols = list(set(dat.columns) - set(self.cols_to_drop))
        return dat.loc[:, lstcols]

    def fit(self, X, y=None):
        dat = X.copy()
        self.lstRemCols = self._delSameCols(dat)
        return self


class DataFrameTransf:
    def __init__(self):
        """
        Class that drops unwanted duplicate rows
        """

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.drop_duplicates()

        return X


class ColumnTransf(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Class to change column types and replace "," by "." in numbers

        """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.copy()
        cols_wrong_type = ["v2", "v3", "v8"]
        for col in cols_wrong_type:
            X.loc[:, col] = pd.to_numeric(
                X.loc[:, col].astype(str).apply(lambda x: x.replace(",", ".")),
                errors="coerce",
            )

        return X