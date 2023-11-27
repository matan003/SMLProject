import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Because of colinearity, qda is not good.

data = pd.read_csv('../Data/training_data.csv')

X = data.drop('increase_stock', axis = 1)
y = data['increase_stock']

lda_model = LinearDiscriminantAnalysis()

# Fit the model on the training data
lda_model.fit(X, y)

# Predict on the training data
y_pred = lda_model.predict(X)

# Calculate accuracy on the training data
training_accuracy = accuracy_score(y, y_pred)
print("Training Accuracy: ", training_accuracy)

kfold = KFold(n_splits = 500, random_state = 42, shuffle = True)

scores = cross_val_score(lda_model, X, y, cv = kfold)

print("Accuracy scores for each fold: ", scores)
print("Average cross-validation score: ", np.mean(scores))

# Feature engineering
morning_peak_start = 6
morning_peak_end = 9
evening_peak_start = 16
evening_peak_end = 19

def is_peak_hour(hour, weekday):
    is_weekday_peak = weekday

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

