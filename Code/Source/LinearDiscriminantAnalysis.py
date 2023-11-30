import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# Because of colinearity, qda is not good.

data = pd.read_csv('../Data/training_data_engineer_with_is_peak_hour.csv')

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

kfold = KFold(n_splits = 5, random_state = 42, shuffle = True)

scores = cross_val_score(lda_model, X, y, cv = kfold)

print("Accuracy scores for each fold: ", scores)
print("Average cross-validation score: ", np.mean(scores))

# Exhaustive feature selection

lda = LinearDiscriminantAnalysis()
kf = KFold(n_splits = 2, random_state = 42, shuffle = True)

#efs = EFS(lda,
#          min_features = 2,
#          max_features = 5,
#          scoring = 'accuracy',
#          print_progress = True,
#          cv = kf
#          )

#efs = efs.fit(X, y)

#print('Best accuracy score: %.2f' % efs.best_score_)
#print('Best feature combination:', efs.best_feature_names_)
