import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree


def main():
    # Setting data base url
    redWine_DataSet_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    # whiteWine_DataSet_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

    # reading csv using pandas
    redData = pd.read_csv(redWine_DataSet_url, sep=';')
    # whiteData = pd.read_csv(whiteWine_DataSet_url, sep=';')

    # print(redData.head())
    # print(whiteData.head())

    # split quality from the rest of the data
    y = redData.quality
    X = redData.drop('quality', axis=1)

    # prepare train data and test data
    # test_size=0.2 means that 80% of data is used to train model then 20% is used to test it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # print(X_train.head())

    # creating and training Decision Tree
    DTC = tree.DecisionTreeClassifier()
    DTC.fit(X_train, y_train)

    # checking confidance of predicted quality
    confidence = DTC.score(X_test, y_test)
    print("\nThe confidence score:\n")
    print(confidence)

    pass
